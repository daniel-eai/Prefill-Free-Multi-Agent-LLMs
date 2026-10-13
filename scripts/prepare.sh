#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

"$PYTHON" -m heterofold download-models llama8 qwen4 ministral14
"$PYTHON" -m heterofold prepare-data
