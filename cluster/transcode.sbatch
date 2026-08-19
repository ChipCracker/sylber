#!/bin/bash
#SBATCH --job-name=sylber2-transcode
#SBATCH --partition=p0
#SBATCH --qos=gpubasic
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/nfs1/scratch/staff/witzlch/sylber2/logs/transcode-%j.out
set -e
source "${SLURM_SUBMIT_DIR:-.}/cluster/common.sh" 2>/dev/null || source "$(dirname "$0")/common.sh"
exec python scripts/transcode_to_flac.py --workers 16 "$@"
