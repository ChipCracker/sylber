#!/bin/bash
#SBATCH --job-name=sylber2-stage1
#SBATCH --partition=p4,p2,p1
#SBATCH --qos=gpuultimate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
set +e
export SYLBER2_WATCHDOG=1800
# locate common.sh: sbatch copies the script to a spool dir, so $0 is useless there
source "${SLURM_SUBMIT_DIR:-.}/cluster/common.sh" 2>/dev/null || source "$(dirname "$0")/common.sh"
OUT="$SYLBER2_ROOT/outputs/${SYLBER2_RUN_NAME:-stage1}"
mkdir -p "$OUT" && cd "$OUT"
for attempt in 1 2 3 4 5; do
  LATEST=$(ls -t "$OUT"/run/lightning_logs/*/checkpoints/last.ckpt 2>/dev/null | head -1)
  echo "=== attempt $attempt (resume: ${LATEST:-none}) ==="
  if python -m sylber2.train_content --config-name content_stage1 hydra.run.dir="$OUT/run" name="$OUT/run" ${LATEST:+resume_ckpt="$LATEST"} "$@"; then
    echo "=== training finished ==="
    TRAINING_OK=1
    break
  fi
  echo "=== attempt $attempt failed, retrying from latest checkpoint ==="
  sleep 30
done
# a job that never finished must not report COMPLETED
[ -n "$TRAINING_OK" ] || exit 1
