#!/bin/bash
#SBATCH --job-name=sylber2-cycle2
#SBATCH --partition=p4,p2,p1
#SBATCH --qos=gpuultimate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
set -e
# locate common.sh: sbatch copies the script to a spool dir, so $0 is useless there
source "${SLURM_SUBMIT_DIR:-.}/cluster/common.sh" 2>/dev/null || source "$(dirname "$0")/common.sh"
OUT="$SYLBER2_ROOT/outputs/cycle2"
mkdir -p "$OUT" && cd "$OUT"
python -m sylber2.train_vocoder --config-name vocoder_cycle2 \
    hydra.run.dir="$OUT/run" name="$OUT/run" model.content_ckpt=$SYLBER2_ROOT/outputs/stage4/run/lightning_logs/version_0/checkpoints/last.ckpt warm_start_ckpt=$SYLBER2_ROOT/outputs/cycle1/run/lightning_logs/version_0/checkpoints/last.ckpt "$@"
