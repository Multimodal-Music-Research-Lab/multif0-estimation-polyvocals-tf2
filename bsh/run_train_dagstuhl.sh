#!/usr/bin/env bash
# run_train_dagstuhl.sh
#
# Train model3 (Late/Deep) on the Dagstuhl-only subset.
# Run after bsh/run_prepare_dagstuhl.sh.
#
# Usage: bash bsh/run_train_dagstuhl.sh

set -euo pipefail
mkdir -p ~/logs
cd "$(dirname "$0")/.."
PYTHONPATH=. uv run python experiments/train_dast.py \
    --save-key dast_model3_dagstuhl_test \
    --data-splits-file data_splits_dagstuhl.json \
    2>&1 | tee ~/logs/train_dast_dagstuhl_$(date +%Y%m%d_%H%M%S).log
