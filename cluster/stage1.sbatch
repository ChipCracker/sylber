#!/bin/bash
#SBATCH --job-name=sylber2-stage1
#SBATCH --partition=p4,p2,p1
#SBATCH --qos=gpuultimate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=%x-%j.out
set -e
source "$(dirname "$0")/common.sh"
OUT="$SYLBER2_ROOT/outputs/stage1"
mkdir -p "$OUT" && cd "$OUT"
python -m sylber2.train_content --config-name content_stage1 \
    hydra.run.dir="$OUT/run" name="$OUT/run"  "$@"
