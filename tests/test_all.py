"""Shape/smoke tests for the Sylber 2.0 implementation (CPU, tiny inputs)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

PASS = []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  ok: {name}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  FAIL: {name}: {e}")
        sys.exit(1)


# ---------------------------------------------------------------- segmentation
def test_segmentation():
    from sylber2.segmentation import (get_segments_greedy, greedy_segment,
                                      refine_short_segments, detect_boundaries,
                                      boundaries_to_segments,
                                      segments_to_boundary_targets)
    rng = np.random.RandomState(0)
    # three well-separated clusters of 10/6/9 frames
    protos = rng.randn(3, 16) * 5
    states = np.concatenate([
        protos[0] + 0.1 * rng.randn(10, 16),
        protos[1] + 0.1 * rng.randn(6, 16),
        protos[2] + 0.1 * rng.randn(9, 16)])
    segs = get_segments_greedy(states, merge_threshold=0.6, refine_threshold=0.5)
    assert segs[0][0] == 0 and segs[-1][1] == 25, segs
    # full coverage, no overlaps
    for a, b in zip(segs[:-1], segs[1:]):
        assert a[1] == b[0]
    assert 2 <= len(segs) <= 5, segs

    # short segment merging: make a 2-frame segment similar to its neighbor
    states2 = np.concatenate([protos[0] + 0.05 * rng.randn(8, 16),
                              protos[0] * 1.02 + 0.05 * rng.randn(2, 16),
                              protos[1] + 0.05 * rng.randn(8, 16)])
    raw = np.array([[0, 8], [8, 10], [10, 18]])
    refined = refine_short_segments(states2, raw, refine_threshold=0.5)
    assert len(refined) == 2 and refined[0][1] == 10, refined

    probs = np.zeros(50); probs[[10, 25, 40]] = 0.9; probs[24] = 0.5
    b = detect_boundaries(probs)
    assert set(b.tolist()) == {10, 25, 40}, b
    segs = boundaries_to_segments(b, 50)
    assert segs.tolist() == [[0, 10], [10, 25], [25, 40], [40, 50]]
    t = segments_to_boundary_targets(segs, 50)
    assert t.sum() == 3 and t[10] == 1


# ---------------------------------------------------------------- content model
def small_encoder_kwargs():
    return {}


def test_sylber2_stages():
    from sylber2.models.sylber2_model import Sylber2
    wav = torch.randn(2, 16000)
    for stage in (1, 2, 3, 4):
        model = Sylber2(stage=stage, load_pretrained=False)
        out = model(student_input=wav, teacher_input=wav)
        assert torch.isfinite(out["distillation_loss"]), (stage, out)
        if stage >= 3:
            assert "boundary_loss" in out and torch.isfinite(out["boundary_loss"])
        loss = sum(v for k, v in out.items() if k.endswith("_loss"))
        loss.backward()
        grads = [p.grad for p in model.student.backbone.parameters() if p.grad is not None]
        assert len(grads) > 0, f"stage {stage}: no grads on student backbone"
        if stage >= 3:
            bg = [p.grad for p in model.student.boundary_detector.parameters()
                  if p.grad is not None]
            assert len(bg) > 0, f"stage {stage}: no grads on boundary detector"
        # teacher must be frozen
        if stage > 1:
            assert all(not p.requires_grad for p in model.teacher_backbone.parameters())
        print(f"    stage {stage}: distill={out['distillation_loss'].item():.4f}")


def test_inference_segment():
    from sylber2.models.sylber2_model import Sylber2
    model = Sylber2(stage=4, load_pretrained=False)
    wav = torch.randn(1, 16000)
    res = model.segment(wav, use_boundary_detector=True)
    r = res[0]
    assert r["frames"].shape[0] == 49
    if len(r["segments"]):
        assert r["content_embedding"].shape[-1] == 64
    res2 = model.segment(wav, use_boundary_detector=False)
    assert len(res2[0]["segments"]) >= 1


# ---------------------------------------------------------------- acoustic
def test_acoustic_alignment():
    from sylber2.models.acoustic_encoder import AcousticEncoder
    enc = AcousticEncoder(load_pretrained=False, hidden_size=768)
    wav24 = torch.randn(2, 72000)   # 3 s at 24 kHz
    frames = enc.forward_frames(wav24)
    assert frames.shape[:2] == (2, 149), frames.shape  # matches 16k HuBERT frames
    segs = [np.array([[0, 50], [50, 100], [100, 149]]), np.array([[0, 149]])]
    embs = enc(wav24, segs)
    assert embs[0].shape == (3, 64) and embs[1].shape == (1, 64)


# ---------------------------------------------------------------- vocoder
def test_vocoder():
    from sylber2.models.vocoder import SylberVocoder, make_frame_inputs, WSegPE
    voc = SylberVocoder(dim=128, intermediate_dim=256, num_layers=2)
    B, L = 2, 149
    content_embs = [torch.randn(3, 64), torch.randn(1, 64)]
    acoustic_embs = [torch.randn(3, 64), torch.randn(1, 64)]
    segs = [np.array([[0, 50], [50, 100], [100, 149]]), np.array([[0, 149]])]
    c, a, p = make_frame_inputs(content_embs, acoustic_embs, segs, L)
    assert c.shape == (2, 149, 64) and p.max() <= 1.0
    assert torch.allclose(c[0, 0], content_embs[0][0])
    assert torch.allclose(c[0, 60], content_embs[0][1])
    audio = voc(c, a, p)
    assert audio.shape == (2, 149 * 480), audio.shape
    audio.sum().backward()
    pe = WSegPE(64)
    out = pe(torch.rand(2, 10))
    assert out.shape == (2, 10, 64)

    n_params = sum(p.numel() for p in SylberVocoder().parameters())
    print(f"    full vocoder params: {n_params/1e6:.1f}M")


# ---------------------------------------------------------------- discriminators
def test_gan_losses():
    from sylber2.models.discriminators import (MultiPeriodDiscriminator,
        MultiResolutionDiscriminator, discriminator_hinge_loss,
        generator_hinge_loss, feature_matching_loss, MelSpecLoss)
    real, fake = torch.randn(2, 24000), torch.randn(2, 24000)
    mpd, mrd = MultiPeriodDiscriminator(), MultiResolutionDiscriminator()
    r, f = mpd(real), mpd(fake)
    d = discriminator_hinge_loss(r, f); g = generator_hinge_loss(f)
    fm = feature_matching_loss(r, f)
    assert all(torch.isfinite(x) for x in (d, g, fm))
    r, f = mrd(real), mrd(fake)
    assert torch.isfinite(discriminator_hinge_loss(r, f))
    mel = MelSpecLoss()(fake, real)
    assert torch.isfinite(mel)


# ---------------------------------------------------------------- synthesis e2e
def test_synthesis_model():
    from sylber2.models.sylber2_model import Sylber2
    from sylber2.models.synthesis_model import SynthesisModel
    from sylber2.models.acoustic_encoder import AcousticEncoder
    from sylber2.models.vocoder import SylberVocoder
    content = Sylber2(stage=4, load_pretrained=False)
    model = SynthesisModel(
        content_encoder=content,
        acoustic_encoder=AcousticEncoder(load_pretrained=False),
        vocoder=SylberVocoder(dim=128, intermediate_dim=256, num_layers=2),
        mean_pool_acoustics_prob=0.5, shuffle_acoustics_prob=0.3)
    model.train()
    wav16, wav24 = torch.randn(2, 48000), torch.randn(2, 72000)
    audio, segs = model(wav16, wav24)
    assert audio.shape[0] == 2 and audio.shape[1] % 480 == 0, audio.shape
    audio.sum().backward()
    # frozen content backbone must have no grads; content_proj must have grads
    assert all(p.grad is None for p in model.content_model.student.backbone.parameters())
    cp = [p.grad for p in model.content_model.student.content_proj.parameters()
          if p.grad is not None]
    ap = [p.grad for p in model.acoustic_encoder.parameters() if p.grad is not None]
    vp = [p.grad for p in model.vocoder.parameters() if p.grad is not None]
    assert cp and ap and vp, (len(cp), len(ap), len(vp))


# ---------------------------------------------------------------- augmentation
def test_augment():
    from sylber2.augment import ContentAugment, mix_at_snr, random_formant_perturb
    wav = np.sin(2 * np.pi * 220 * np.arange(16000 * 2) / 16000).astype(np.float32)
    aug = ContentAugment(sr=16000)  # no dirs -> formant/white noise only
    out = aug(wav)
    assert out.shape == wav.shape and np.isfinite(out).all()
    mixed = mix_at_snr(wav, np.random.randn(len(wav)).astype(np.float32), 20)
    assert mixed.shape == wav.shape
    fp = random_formant_perturb(wav, 16000)
    assert fp.shape == wav.shape


if __name__ == "__main__":
    torch.manual_seed(0); np.random.seed(0)
    print("segmentation:"); check("segmentation", test_segmentation)
    print("augment:"); check("augment", test_augment)
    print("sylber2 stages:"); check("stages", test_sylber2_stages)
    print("inference:"); check("inference", test_inference_segment)
    print("acoustic:"); check("acoustic", test_acoustic_alignment)
    print("vocoder:"); check("vocoder", test_vocoder)
    print("gan:"); check("gan", test_gan_losses)
    print("synthesis:"); check("synthesis", test_synthesis_model)
    print(f"\nALL {len(PASS)} TESTS PASSED")
