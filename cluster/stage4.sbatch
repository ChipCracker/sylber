#!/bin/bash
#SBATCH --job-name=sylber2-stage4
#SBATCH --partition=p4,p2,p1
#SBATCH --qos=gpuultimate
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
set +e
export SYLBER2_WATCHDOG=1800
# locate common.sh: sbatch copies the script to a spool dir, so $0 is useless there
source "${SLURM_SUBMIT_DIR:-.}/cluster/common.sh" 2>/dev/null || source "$(dirname "$0")/common.sh"
OUT="$SYLBER2_ROOT/outputs/${SYLBER2_RUN_NAME:-stage4}"
mkdir -p "$OUT" && cd "$OUT"
for attempt in $(seq 1 20); do
  LATEST=$(ls -t "$OUT"/run/lightning_logs/*/checkpoints/last.ckpt 2>/dev/null | head -1)
  echo "=== attempt $attempt (resume: ${LATEST:-none}) ==="
  if srun python -m sylber2.train_content --config-name content_stage4 hydra.run.dir="$OUT/run" name="$OUT/run" model_ckpt=${SYLBER2_PREV_CKPT:-$(ls -t $SYLBER2_ROOT/outputs/stage3/run/lightning_logs/*/checkpoints/last.ckpt | head -1)} devices=4 strategy=ddp_find_unused_parameters_true ${LATEST:+resume_ckpt="$LATEST"} "$@"; then
    echo "=== training finished ==="
    TRAINING_OK=1
    break
  fi
  echo "=== attempt $attempt failed, retrying from latest checkpoint ==="
  sleep 30
done
# a job that never finished must not report COMPLETED
[ -n "$TRAINING_OK" ] || exit 1
