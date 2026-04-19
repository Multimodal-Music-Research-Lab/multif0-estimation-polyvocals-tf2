#!/usr/bin/env bash
# run_evaluation.sh
#
# Evaluate the trained dAST model on train/val/test splits.
# Run after bsh/run_train_full.sh.
#
# Usage: bash bsh/run_evaluation.sh

set -euo pipefail
mkdir -p ~/logs
cd "$(dirname "$0")/.."
CUDA_VISIBLE_DEVICES=1,2,3 PYTHONPATH=. uv run python experiments/evaluate_dast.py \
    --save-key dast_model3_v1 \
    --data-splits-file data_splits.json \
    --splits train,validate,test \
    2>&1 | tee ~/logs/evaluate_dast_$(date +%Y%m%d_%H%M%S).log
