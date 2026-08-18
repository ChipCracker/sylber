#!/bin/bash
# One-time environment setup on kiz0 (run on the login node; internet is direct).
set -e
# Auto-detect the scratch root on kiz0 (students vs staff)
if [ -z "$SYLBER2_ROOT" ]; then
    for base in /nfs1/scratch/students/$USER /nfs1/scratch/staff/$USER; do
        if [ -d "$base" ]; then SYLBER2_ROOT="$base/sylber2"; break; fi
    done
    SYLBER2_ROOT="${SYLBER2_ROOT:-$HOME/sylber2}"
fi
export SYLBER2_ROOT
mkdir -p "$SYLBER2_ROOT"
cd "$SYLBER2_ROOT"

# uv (static binary, no root needed)
if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# repo
if [ ! -d sylber ]; then
    git clone https://github.com/ChipCracker/sylber.git sylber
fi
cd sylber && git fetch && git checkout sylber2 && git pull --ff-only && cd ..

# venv with CUDA-enabled torch (H200 = sm_90, any recent cu12x wheel works)
if [ ! -d venv ]; then
    uv venv --python 3.12 venv
fi
uv pip install --python venv/bin/python -r sylber/requirements-sylber2.txt
venv/bin/python - <<'PY'
import torch, transformers, lightning, parselmouth, datasets
print("torch", torch.__version__, "cuda build:", torch.version.cuda)
PY
echo "Setup complete under $SYLBER2_ROOT"
