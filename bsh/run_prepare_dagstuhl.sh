#!/usr/bin/env bash
# run_prepare_dagstuhl.sh
#
# Extract HCQT features for Dagstuhl-only mixture pairs and write
# data_splits_dagstuhl.json.  Run this to validate the pipeline
# end-to-end before committing to full extraction.
#
# Requires mixture_pairs.csv to exist (run bsh/run_generate_pairs.sh first).
#
# Usage: bash bsh/run_prepare_dagstuhl.sh

set -euo pipefail
mkdir -p ~/logs
cd "$(dirname "$0")/.."
PYTHONPATH=. uv run python experiments/prepare_dast_mixture_features.py \
    --datasets dagstuhl_manual dagstuhl_non_manual \
    --output-splits-file data_splits_dagstuhl.json \
    --n-jobs 4 \
    2>&1 | tee ~/logs/prepare_mixture_dagstuhl_$(date +%Y%m%d_%H%M%S).log
