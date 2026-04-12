#!/usr/bin/env bash
# run_train_full.sh
#
# Train model3 (Late/Deep) on the full dAST dataset (all splits).
# Run after bsh/run_prepare_full.sh.
#
# Usage: bash bsh/run_train_full.sh

set -euo pipefail
mkdir -p ~/logs
cd "$(dirname "$0")/.."
PYTHONPATH=. uv run python experiments/train_dast.py \
    --save-key dast_model3_v1 \
    --data-splits-file data_splits.json \
    2>&1 | tee ~/logs/train_dast_$(date +%Y%m%d_%H%M%S).log
