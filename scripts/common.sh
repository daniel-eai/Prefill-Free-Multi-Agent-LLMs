#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
GPUS="${GPUS:-0}"
MAPS="${MAPS:-maps}"
RESULTS="${RESULTS:-results}"
export HETEROFOLD_DATA="${HETEROFOLD_DATA:-data}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
mkdir -p logs

LONG_CONTEXT="qasper hotpotqa locomo quality"
SHORT_CONTEXT="arc mmlu winogrande hellaswag gsm8k"

model_gb() {
  case "$1" in
    qwen4) echo 8 ;; llama8) echo 16 ;; ministral14) echo 28 ;;
    *) echo 16 ;;
  esac
}

gpus_per_job() {
  if [[ -n "${GPUS_PER_JOB:-}" ]]; then echo "$GPUS_PER_JOB"; return; fi
  local name="$1" need=14 first total
  for model in ${name//_to_/ }; do need=$((need + $(model_gb "$model"))); done
  first="${GPUS%%,*}"
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$first" | head -1)
  if (( need * 1024 > total )); then echo 2; else echo 1; fi
}

gpu_groups() {
  local per="$1" ids group=() id
  IFS=',' read -ra ids <<< "$GPUS"
  for id in "${ids[@]}"; do
    group+=("$id")
    if (( ${#group[@]} == per )); then
      (IFS=','; echo "${group[*]}"); group=()
    fi
  done
}
