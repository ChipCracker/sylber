#!/bin/bash
#SBATCH --job-name=sylber2-stage4
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
OUT="$SYLBER2_ROOT/outputs/stage4"
mkdir -p "$OUT" && cd "$OUT"
python -m sylber2.train_content --config-name content_stage4 \
    hydra.run.dir="$OUT/run" name="$OUT/run" model_ckpt=$SYLBER2_ROOT/outputs/stage3/run/lightning_logs/version_0/checkpoints/last.ckpt "$@"
