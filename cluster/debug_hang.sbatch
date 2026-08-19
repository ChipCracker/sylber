#!/bin/bash
#SBATCH --job-name=sylber2-debug
#SBATCH --partition=p4
#SBATCH --qos=advanced
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:50:00
#SBATCH --output=%x-%j.out
set -e
source "${SLURM_SUBMIT_DIR:-.}/cluster/common.sh" 2>/dev/null || source "$(dirname "$0")/common.sh"
export SYLBER2_WATCHDOG=240
echo "===== TEST 1: pure dataloader on GPU node ====="
timeout 420 python tests/repro_dataloader.py --workers 12 --formant-prob 0.3 2>&1 | grep -vE "layer3|Praat" | tail -5 || echo "TEST1 TIMEOUT/FAIL"
echo "===== TEST 2: real training, 20 steps, real data ====="
OUT=$SYLBER2_ROOT/outputs/debug_$SLURM_JOB_ID
timeout 1500 python -m sylber2.train_content --config-name content_stage1 \
    max_steps=20 checkpoint_every_steps=20 val_check_interval=19 limit_val_batches=1 \
    data.batch_size=24 data.num_workers=12 data.dummy_len=2000 model.warmup_steps=5 \
    hydra.run.dir="$OUT" name="$OUT" 2>&1 | grep -vE "layer3|Praat" | tail -8 || echo "TEST2 TIMEOUT/FAIL"
echo "===== DEBUG DONE ====="
