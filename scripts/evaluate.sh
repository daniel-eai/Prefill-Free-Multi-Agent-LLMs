#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

NAME="${1:?usage: evaluate.sh <sender>_to_<receiver>|<receiver> [benchmark ...]}"
shift
BENCHMARKS="${*:-$LONG_CONTEXT $SHORT_CONTEXT}"

if [[ "$NAME" == *_to_* ]]; then
  RUN="${RUN:-heterofold}"
  MAP="${MAP:-$MAPS/$NAME/heterofold.pt}"
  [[ -f "$MAP" ]] || { echo "map $MAP not found; run scripts/calibrate.sh $NAME" >&2; exit 1; }
  MODE=(--pair "$NAME" --map "$MAP")
else
  RUN="native"
  MODE=(--receiver "$NAME" --native)
fi
OUT="$RESULTS/$NAME/$RUN"

mapfile -t GROUPS_ < <(gpu_groups "$(gpus_per_job "$NAME")")
JOBS=${#GROUPS_[@]}
(( JOBS > 0 )) || { echo "not enough GPUs in GPUS=$GPUS for $NAME" >&2; exit 1; }
echo "[evaluate.sh] $NAME ($RUN): $BENCHMARKS | $JOBS job(s) on GPU group(s): ${GROUPS_[*]}"

PIDS=()
for index in "${!GROUPS_[@]}"; do
  LOG="logs/evaluate_${NAME}_${RUN}_shard${index}of${JOBS}.log"
  CUDA_VISIBLE_DEVICES="${GROUPS_[$index]}" "$PYTHON" -m heterofold evaluate \
    "${MODE[@]}" --benchmark $BENCHMARKS --shard "$index/$JOBS" --out "$OUT" \
    > "$LOG" 2>&1 &
  PIDS+=($!)
done
FAILED=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=1; done
if (( FAILED )); then
  echo "[evaluate.sh] a shard failed; see logs/evaluate_${NAME}_${RUN}_shard*.log" >&2
  exit 1
fi
"$PYTHON" -m heterofold summarize --results "$RESULTS" --pair "$NAME" --paper
