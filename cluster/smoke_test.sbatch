#!/bin/bash
#SBATCH --job-name=sylber2-smoke
#SBATCH --partition=p4,p2,p1
#SBATCH --qos=basic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# End-to-end smoke test: unit tests on GPU + a few real training steps of
# every stage/cycle through the actual Hydra entry points + inference.
set -e
# locate common.sh: sbatch copies the script to a spool dir, so $0 is useless there
source "${SLURM_SUBMIT_DIR:-.}/cluster/common.sh" 2>/dev/null || source "$(dirname "$0")/common.sh"
OUT="$SYLBER2_ROOT/outputs/smoke_$SLURM_JOB_ID"
mkdir -p "$OUT" && cd "$OUT"

echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
python "$REPO/tests/test_all.py"

SMOKE="max_steps=30 checkpoint_every_steps=30 val_check_interval=25 data.batch_size=8 data.num_workers=8 data.dummy_len=2000 devices=1"

echo "=== stage 1 ==="
python -m sylber2.train_content --config-name content_stage1 $SMOKE \
    model.warmup_steps=5 hydra.run.dir="$OUT/stage1" name="$OUT/stage1"
CKPT1=$(find "$OUT/stage1" -name "last.ckpt" | head -1)

echo "=== stage 2 ==="
python -m sylber2.train_content --config-name content_stage2 $SMOKE \
    model.warmup_steps=5 model_ckpt="$CKPT1" hydra.run.dir="$OUT/stage2" name="$OUT/stage2"
CKPT2=$(find "$OUT/stage2" -name "last.ckpt" | head -1)

echo "=== stage 3 ==="
python -m sylber2.train_content --config-name content_stage3 $SMOKE \
    model.warmup_steps=5 model_ckpt="$CKPT2" hydra.run.dir="$OUT/stage3" name="$OUT/stage3"
CKPT3=$(find "$OUT/stage3" -name "last.ckpt" | head -1)

echo "=== stage 4 ==="
python -m sylber2.train_content --config-name content_stage4 $SMOKE \
    model.warmup_steps=5 model_ckpt="$CKPT3" hydra.run.dir="$OUT/stage4" name="$OUT/stage4"
CKPT4=$(find "$OUT/stage4" -name "last.ckpt" | head -1)

echo "=== vocoder cycle 1 ==="
python -m sylber2.train_vocoder --config-name vocoder_cycle1 \
    max_steps=20 checkpoint_every_steps=20 val_check_interval=15 \
    data.batch_size=4 data.num_workers=8 data.dummy_len=500 devices=1 \
    model.warmup_steps=5 model.content_ckpt="$CKPT4" \
    hydra.run.dir="$OUT/cycle1" name="$OUT/cycle1"
CKPT_V1=$(find "$OUT/cycle1" -name "last.ckpt" | head -1)

echo "=== vocoder cycle 2 (with WavLM perceptual loss) ==="
python -m sylber2.train_vocoder --config-name vocoder_cycle2 \
    max_steps=10 checkpoint_every_steps=10 val_check_interval=8 \
    data.batch_size=4 data.num_workers=8 data.dummy_len=200 devices=1 \
    model.warmup_steps=2 model.content_ckpt="$CKPT4" warm_start_ckpt="$CKPT_V1" \
    hydra.run.dir="$OUT/cycle2" name="$OUT/cycle2"
CKPT_V2=$(find "$OUT/cycle2" -name "last.ckpt" | head -1)

echo "=== inference ==="
python - <<PY
import glob, soundfile as sf, numpy as np, os
from sylber2.inference import Segmenter2
wavs = sorted(glob.glob(os.environ["SYLBER2_DATA"] + "/audio/fleurs/*/*.flac"))
seg = Segmenter2("$CKPT4", synthesis_ckpt="$CKPT_V2", device="cuda")
out = seg(wav_file=wavs[0])
print("segments:", out["segments"][:5], "... total", len(out["segments"]))
print("content_embedding:", None if out["content_embedding"] is None else out["content_embedding"].shape)
audio, segs = seg.resynthesize(wav_file=wavs[0])
sf.write("$OUT/resynth_sample.wav", audio, 24000)
print("resynth ok:", audio.shape, "->", "$OUT/resynth_sample.wav")
PY
echo "=== SMOKE TEST PASSED ==="
