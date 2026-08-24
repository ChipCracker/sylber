# Sylber 2.0 Reimplementation — Assumptions

The paper (arXiv:2601.22306) leaves several details unspecified. This file
documents every assumption made in this implementation.

1. **Teacher targets**: layer-8 features, L2-normalized. For stages 2-4 the
   segment mean is computed first, then L2-normalized. Stage 1 normalizes
   frame-wise. (Paper: "teacher targets are L2-normalized"; averaging order
   not specified.)
2. **mHuBERT init**: `HubertModel.from_pretrained("utter-project/mHuBERT-147",
   num_hidden_layers=9)` — layers 0-8 loaded, then layers 6-8 Xavier-reinitialized
   ("last three layers are randomly reinitialized").
3. **Boundary detector input**: last-layer (9) student/teacher frames,
   detached for the BCE loss so boundary training does not affect the
   distillation objective. BCE coefficient = 1.0 (not specified).
4. **Stage 4 teacher**: frozen copy of student backbone AND boundary detector
   from the end of stage 3; teacher segments = peak detection on the teacher
   boundary probabilities; BCE targets = those teacher segments.
5. **Residual FC layers** ("reducing the dimension to 64"): 2 pre-LN residual
   blocks (768→768) + a linear 768→64. Count not specified (TTS appendix uses
   6 blocks at dim 512 for its RF head, suggesting order-of-magnitude).
6. **Acoustic encoder transformers**: hidden 768, 12 heads, FFN 3072, with
   HuBERT-style convolutional positional embedding (dims not specified).
   CNN weights from WavLM-Large; only the 2nd conv layer's stride changes 2→3
   (kernel weights kept).
7. **Vocoder size**: dim 1024, FFN 4096, 12 ConvNeXt blocks, n_fft 1920,
   hop 480 → 103M params ("only 100M parameters"). iSTFT via windowed
   overlap-add with 'same' padding (output = 50 Hz × 480).
8. **wSegPE**: 64-d, concatenated (not added); template linearly interpolated
   over relative position p ∈ [0,1] with 11 entries (paper specifies 11 and
   interpolation; dimension/concat is our choice).
9. **Adversarial losses**: "same losses as Vocos" → multi-period (HiFi-GAN
   periods 2,3,5,7,11) + multi-resolution spectrogram discriminator, hinge
   loss, feature matching (coef 1), log-mel L1 (coef 45, Vocos default).
10. **Perceptual loss**: L1 on WavLM-Large hidden states 0 (CNN), 3, 6, 9, 12,
    averaged; weight 1.0; audio resampled 24→16 kHz inside the loss.
11. **Augmentation probabilities**: formant 0.3, env-noise 0.2 XOR speech-clip
    0.05, white noise 0.3 (all per paper); RIR probability not given → 0.25.
    SNR ranges (not given): env-noise U(5,25) dB, speech clips U(5,20) dB,
    white noise U(10,40) dB.
12. **Formant perturbation**: NANSY/ContentVec-style Praat "Change gender":
    formant ratio U(1,1.4)^±1, pitch-median ratio U(1,2)^±1, pitch-range
    ratio U(1,1.5)^±1, duration 1.0 (length-preserving).
13. **"Perturbing Audio" in cycle training** (Table 7): environmental noise or
    RIR applied to the acoustic-encoder input and the reconstruction target
    (the 16 kHz content input always stays original).
14. **Vocoder optimizer schedule**: cosine from max LR (Table 7) with 5000
    warmup steps and min factor 0.01 per cycle (warmup not specified).
15. **LR schedule (content)**: warmup then constant (min_factor=1), matching
    the original Sylber configs; Table 6 only specifies warmup steps.
16. **Emilia/MLS language balancing**: each language = one manifest source
    with weight 1; FLEURS = one source with weight 2 ("twice the sampling
    probability of an individual language").
17. **Validation**: small held-back sampling of the training sources without
    augmentation (the paper does not describe a validation protocol).
18. **Stage-1 target centering (necessary deviation)**: the paper's stage-1
    recipe (EMA teacher, L2-normalized targets, single student FC head)
    collapsed in our reproduction — after 100k steps every frame produced the
    same direction (pairwise cosine similarity 1.0, greedy segmentation
    degenerated to one segment per clip, stage-2 loss went to 0). We
    therefore subtract a DINO-style running center (decay 0.99) from the
    teacher's layer-8 features before L2 normalization in stage 1 (only);
    stages 2-4 use a frozen teacher and need no centering. `target_sim` is
    logged as a collapse early-warning metric (≈1.0 ⇒ collapsed).
19. **Stages 2-4 operate on instance-normalized teacher features** (both the
    greedy segmentation input and the segment-averaged targets, plus the
    student prediction is L2-normalized in the loss): consistent with the
    stage-1 target regime (#18), and necessary because the raw layer-8
    features carry a dominant shared direction (pairwise cosine ≈ 0.87)
    that would merge whole clips into one segment under the paper's merge
    thresholds.
20. **Vocoder cycles scaled to batch 48** (from the paper's batch 12): same
    total sample count (steps /4: 2M->500k, 100k->25k), learning rate x2
    (sqrt scaling; GAN-conservative), fits H200 memory. Motivated purely by
    wall-clock (~4x); revert to batch 12 to match the paper exactly.
21. **Content v2 (recalibrated thresholds)**: on instance-normalized features
    the paper's merge thresholds [0.5-0.9] yield 12-15 seg/s (measured);
    tau=0.20 reproduces the paper token rate (~5/s). Content v2 retrains
    stages 2-4 from the same stage-1 checkpoint with stage2 [0.20,0.30]
    (refine 0.20) and stage3/4 [0.35,0.50] (refine 0.35). The original
    (fine-grained, ~10 Hz) line is kept and continues through the cycles.
