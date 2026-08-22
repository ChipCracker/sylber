#!/bin/bash
# Sylber 2.0 pipeline orchestrator: runs stage2..cycle4 sequentially,
# resubmitting on walltime/crash (jobs auto-resume from their newest
# checkpoint). Run with nohup on the kiz0 login node:
#   nohup bash cluster/orchestrator.sh > $SYLBER2_ROOT/logs/orchestrator.log 2>&1 &
set -u
source "$(dirname "$0")/common.sh"
STATE_DIR="$SYLBER2_ROOT/outputs/orchestrator"
mkdir -p "$STATE_DIR"
MAX_RESUBMITS=15
PARTITION="${SYLBER2_PARTITION:-p4}"

PHASES=(stage2 stage3 stage4 cycle1 cycle2 cycle3 cycle4)
log() { echo "$(date "+%F %T") $*"; }

ckpt_of() {  # newest last.ckpt of a run name
  ls -t "$SYLBER2_ROOT/outputs/$1"/run/lightning_logs/*/checkpoints/last.ckpt 2>/dev/null | head -1
}

submit_phase() {  # $1 = phase
  local phase="$1" prev="" args=""
  export SYLBER2_RUN_NAME="${phase}_auto"
  unset SYLBER2_PREV_CKPT SYLBER2_CONTENT_CKPT SYLBER2_WARM_CKPT
  case "$phase" in
    stage2) export SYLBER2_PREV_CKPT="$(ckpt_of stage1_v5)";;
    stage3) export SYLBER2_PREV_CKPT="$(ckpt_of stage2_auto)";;
    stage4) export SYLBER2_PREV_CKPT="$(ckpt_of stage3_auto)";;
    cycle1) export SYLBER2_CONTENT_CKPT="$(ckpt_of stage4_auto)";;
    cycle2) export SYLBER2_CONTENT_CKPT="$(ckpt_of stage4_auto)"
            export SYLBER2_WARM_CKPT="$(ckpt_of cycle1_auto)";;
    cycle3) export SYLBER2_CONTENT_CKPT="$(ckpt_of stage4_auto)"
            export SYLBER2_WARM_CKPT="$(ckpt_of cycle2_auto)";;
    cycle4) export SYLBER2_CONTENT_CKPT="$(ckpt_of stage4_auto)"
            export SYLBER2_WARM_CKPT="$(ckpt_of cycle3_auto)";;
  esac
  # sanity: required predecessor must exist
  for v in SYLBER2_PREV_CKPT SYLBER2_CONTENT_CKPT SYLBER2_WARM_CKPT; do
    val="${!v:-unset}"
    if [ "$val" != "unset" ] && [ ! -f "$val" ]; then
      log "FATAL $phase: predecessor checkpoint missing ($v=$val)"; return 1
    fi
  done
  local script="cluster/${phase}.sbatch"
  [[ "$phase" == stage* ]] && script="cluster/${phase}.sbatch"
  cd "$REPO"
  out=$(sbatch --partition="$PARTITION" -J "sylber2-${phase}" "$script" 2>&1)
  log "$phase submitted: $out (prev=${SYLBER2_PREV_CKPT:-} content=${SYLBER2_CONTENT_CKPT:-} warm=${SYLBER2_WARM_CKPT:-})"
  echo "$out" | grep -q "Submitted" || return 1
  return 0
}

# wait until no stage1 job is still running (do not chain off a mid-run ckpt)
while squeue -u "$USER" -h -n sylber2-stage1 2>/dev/null | grep -q .; do
  log "waiting for stage1 to finish..."
  sleep 300
done
[ -f "$(ckpt_of stage1_v5)" ] || { log "FATAL: no stage1_v5 checkpoint"; exit 1; }
log "stage1_v5 checkpoint: $(ckpt_of stage1_v5)"

for phase in "${PHASES[@]}"; do
  done_marker="$STATE_DIR/${phase}.done"
  [ -f "$done_marker" ] && { log "$phase already done"; continue; }
  resubmits=0
  while true; do
    # is a job for this phase running/pending?
    if squeue -u "$USER" -h -n "sylber2-${phase}" 2>/dev/null | grep -q .; then
      sleep 300; continue
    fi
    # finished successfully? (retry loop echoes the marker)
    latest_log=$(ls -t "$REPO"/sylber2-${phase}-*.out 2>/dev/null | head -1)
    if [ -n "$latest_log" ] && grep -q "=== training finished ===" "$latest_log"; then
      touch "$done_marker"; log "$phase COMPLETED"; break
    fi
    if [ -n "$latest_log" ] && grep -q "collapse detected" "$latest_log"; then
      log "FATAL: collapse detected in $phase - stopping orchestration"; exit 2
    fi
    if [ "$resubmits" -ge "$MAX_RESUBMITS" ]; then
      log "FATAL: $phase exceeded $MAX_RESUBMITS resubmits"; exit 3
    fi
    submit_phase "$phase" || { log "FATAL: submit failed for $phase"; exit 4; }
    resubmits=$((resubmits + 1))
    sleep 120
  done
done
log "PIPELINE COMPLETE - all phases done"
