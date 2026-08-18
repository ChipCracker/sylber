#!/bin/bash
# Small data subset for smoke tests: FLEURS (3 languages, ~1 h total),
# a tiny FLEURS-R resynthesis subset, OpenSLR-28 RIRs, and speech clips.
set -e
source "$(dirname "$0")/common.sh"
python scripts/prepare_data.py fleurs   --languages de_de es_419 ko_kr --max-hours 1.5 --split dev
python scripts/prepare_data.py fleurs_r --languages de_de es_419 ko_kr --max-hours 1.0 --split dev
python scripts/prepare_data.py rir
python scripts/prepare_data.py speech_clips --num-clips 500
echo "Smoke data ready under $SYLBER2_DATA"
