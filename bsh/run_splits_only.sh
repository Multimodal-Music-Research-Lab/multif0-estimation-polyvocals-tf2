#!/usr/bin/env bash
# run_splits_only.sh
#
# Regenerate data_splits.json without re-extracting any features.
# Use this whenever mixture_pairs.csv is updated — npy files already on
# disk are scanned and the JSON is rewritten.
#
# Usage: bash bsh/run_splits_only.sh

set -euo pipefail
mkdir -p ~/logs
cd "$(dirname "$0")/.."
PYTHONPATH=. uv run python experiments/prepare_dast_mixture_features.py \
    --splits-only \
    2>&1 | tee ~/logs/splits_only_$(date +%Y%m%d_%H%M%S).log
