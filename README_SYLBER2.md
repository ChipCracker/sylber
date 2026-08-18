# Sylber 2.0 — Reimplementation

Faithful reimplementation of **"Sylber 2.0: A Universal Syllable Embedding"**
(Cho, Lee, Black, Anumanchipalli — [arXiv:2601.22306](https://arxiv.org/abs/2601.22306))
on top of the original Sylber repository. All new code lives in `sylber2/`;
the original Sylber 1.0 code in `sylber/` is untouched.

## What is implemented

| Paper section | Code |
|---|---|
| Content encoder: 9 layers, mHuBERT-147 init, last 3 reinit, student FC head, L2-normalized layer-8 teacher targets | `sylber2/models/content_encoder.py` |
| 4-stage self-distillation (Table 6): frame-wise (EMA teacher) → 2× self-segmentation (frozen teacher, greedy+refine) → boundary-detector stage | `sylber2/models/sylber2_model.py`, `sylber2/training/content_trainer.py` |
| Greedy segmentation + <80 ms refinement (A.1.1), thresholds (A.1.2) | `sylber2/segmentation.py` |
| Boundary detector (3 layers + binary logit, BCE, peak detection) | `content_encoder.py`, `segmentation.py` |
| Acoustic encoder: WavLM-Large CNN (stride 2→3, 24 kHz, 50 Hz), 6 transformer layers, 64-d | `sylber2/models/acoustic_encoder.py` |
| Vocos vocoder: 12 ConvNeXt blocks, iSTFT (24 kHz), wSegPE (11-entry template), ~100M params | `sylber2/models/vocoder.py` |
| GAN training: MPD + MRD, hinge, feature matching, log-mel L1, WavLM perceptual loss (layers 0,3,6,9,12) | `sylber2/models/discriminators.py`, `perceptual.py`, `training/vocoder_trainer.py` |
| 4 training cycles with disentanglement strategies (Table 7): voice/audio perturbation, mean-pool/shuffle acoustics, freezing schedule | `sylber2/models/synthesis_model.py`, `sylber2/configs/vocoder_cycle*.yaml` |
| Augmentation (3.2.1): formant perturbation (Praat), env. noise XOR speech clips, RIR, white noise | `sylber2/augment.py` |
| Multilingual data (3.5, A.1.3): language-balanced Emilia/MLS + FLEURS(×2), 5 s crops; FLEURS-R(×7)/EXPRESSO/Globe/GTSinger, 3 s windows | `sylber2/data/`, `scripts/prepare_data.py` |

Undocumented details filled with documented assumptions: see
`IMPLEMENTATION_NOTES.md`.

## Quick start (kiz0 cluster)

```bash
# 1) one-time setup (login node)
bash cluster/setup_env.sh

# 2) small data subset + smoke test (~30 min GPU)
source cluster/common.sh
bash cluster/prepare_smoke_data.sh
sbatch cluster/smoke_test.sbatch

# 3) full data (large downloads; Emilia/GTSinger are HF-gated -> set HF_TOKEN)
python scripts/prepare_data.py fleurs
python scripts/prepare_data.py mls
python scripts/prepare_data.py emilia
python scripts/prepare_data.py fleurs_r
python scripts/prepare_data.py expresso
python scripts/prepare_data.py globe
python scripts/prepare_data.py gtsinger
python scripts/prepare_data.py noise --run
python scripts/prepare_data.py rir
python scripts/prepare_data.py speech_clips

# 4) full training (sequential stages; each fits on a single 24 GB GPU)
sbatch cluster/stage1.sbatch     # then stage2..4 after each finishes
sbatch cluster/cycle1.sbatch     # then cycle2..4
```

## Inference

```python
from sylber2.inference import Segmenter2
seg = Segmenter2("stage4_last.ckpt", synthesis_ckpt="cycle4_last.ckpt")
out = seg(wav_file="sample.wav")        # segments (~5 Hz), 64-d content embeddings
audio, segments = seg.resynthesize(wav_file="sample.wav")  # 24 kHz reconstruction
```

## Training cost (paper reference)

Every stage fits on one 24 GB GPU (paper: RTX A5000). Iterations:
stages 1-3: 100K (batch 72/50/50), stage 4: 200K (batch 50);
vocoder cycles: 2000K + 2000K + 100K + 100K (batch 12).
Cycles 1-2 dominate the total wall-clock by far.
