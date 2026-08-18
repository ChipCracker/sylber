#!/bin/bash
# Shared environment for Sylber 2.0 jobs on kiz0.
# Auto-detect the scratch root on kiz0 (students vs staff)
if [ -z "$SYLBER2_ROOT" ]; then
    for base in /nfs1/scratch/students/$USER /nfs1/scratch/staff/$USER; do
        if [ -d "$base" ]; then SYLBER2_ROOT="$base/sylber2"; break; fi
    done
    SYLBER2_ROOT="${SYLBER2_ROOT:-$HOME/sylber2}"
fi
export SYLBER2_ROOT
export SYLBER2_DATA="${SYLBER2_DATA:-$SYLBER2_ROOT/data}"
export HF_HOME="${HF_HOME:-$SYLBER2_ROOT/hf_cache}"
export REPO="$SYLBER2_ROOT/sylber"
export VENV="$SYLBER2_ROOT/venv"
export PYTHONUNBUFFERED=1
mkdir -p "$SYLBER2_DATA" "$HF_HOME" "$SYLBER2_ROOT/outputs"
cd "$REPO"
source "$VENV/bin/activate"
