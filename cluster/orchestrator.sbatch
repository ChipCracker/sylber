#!/bin/bash
#SBATCH --job-name=sylber2-orchestrator
#SBATCH --partition=p0
#SBATCH --qos=gpubasic
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=/nfs1/scratch/staff/witzlch/sylber2/logs/orchestrator-%j.out
exec bash "${SLURM_SUBMIT_DIR:-.}/cluster/orchestrator.sh"
