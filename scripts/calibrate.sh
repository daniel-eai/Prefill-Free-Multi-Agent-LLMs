#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

PAIR="${1:?usage: calibrate.sh <sender>_to_<receiver> [calibrate options]}"
shift
OUT="${OUT:-$MAPS/$PAIR}"
GROUP="$(gpu_groups "$(gpus_per_job "$PAIR")" | head -1)"
if [[ -z "$GROUP" ]]; then
  echo "calibrating $PAIR needs $(gpus_per_job "$PAIR") GPUs; set GPUS=a,b" >&2
  exit 1
fi
LOG="logs/calibrate_$(echo "$OUT" | tr '/' '_').log"
echo "[calibrate.sh] $PAIR on GPU(s) $GROUP -> $OUT (log: $LOG)"
CUDA_VISIBLE_DEVICES="$GROUP" "$PYTHON" -m heterofold calibrate \
  --pair "$PAIR" --out "$OUT" "$@" 2>&1 | tee "$LOG" \
  | { grep --line-buffered -v "Loading weights" || true; }
