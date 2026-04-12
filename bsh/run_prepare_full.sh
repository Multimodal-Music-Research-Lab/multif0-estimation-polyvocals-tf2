#!/usr/bin/env bash
# run_prepare_full.sh
#
# Extract HCQT features for all dataset mixture pairs and write
# data_splits.json.  Pairs already extracted are skipped automatically.
#
# Requires mixture_pairs.csv to exist (run bsh/run_generate_pairs.sh first).
#
# Usage: bash bsh/run_prepare_full.sh

set -euo pipefail
mkdir -p ~/logs
cd "$(dirname "$0")/.."
PYTHONPATH=. uv run python experiments/prepare_dast_mixture_features.py \
    --n-jobs 4 \
    2>&1 | tee ~/logs/prepare_mixture_full_$(date +%Y%m%d_%H%M%S).log
