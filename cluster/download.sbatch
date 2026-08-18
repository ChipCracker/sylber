#!/bin/bash
#SBATCH --job-name=sylber2-dl
#SBATCH --partition=p0
#SBATCH --qos=gpubasic
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=/nfs1/scratch/staff/witzlch/sylber2/logs/slurm-dl-%j.out
# Generic CPU-only data-download job.
# Usage: sbatch [-J sylber2-dl-<name>] cluster/download.sbatch <subcommand> [args...]
#   e.g. sbatch -J sylber2-dl-fleurs cluster/download.sbatch fleurs --resume
set -e
source "${SLURM_SUBMIT_DIR:-.}/cluster/common.sh" 2>/dev/null || source "$(dirname "$0")/common.sh"
echo "== download job: $* (node $(hostname)) =="
exec python scripts/prepare_data.py "$@"
