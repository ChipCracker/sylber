#!/bin/bash
# Shared environment for Sylber 2.0 jobs on kiz0.
export SYLBER2_ROOT="${SYLBER2_ROOT:-/nfs1/scratch/$USER/sylber2}"
export SYLBER2_DATA="${SYLBER2_DATA:-$SYLBER2_ROOT/data}"
export HF_HOME="${HF_HOME:-$SYLBER2_ROOT/hf_cache}"
export REPO="$SYLBER2_ROOT/sylber"
export VENV="$SYLBER2_ROOT/venv"
export PYTHONUNBUFFERED=1
mkdir -p "$SYLBER2_DATA" "$HF_HOME" "$SYLBER2_ROOT/outputs"
cd "$REPO"
source "$VENV/bin/activate"
