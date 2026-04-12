#!/usr/bin/env bash
# run_generate_pairs.sh
#
# Generate mixture_pairs.csv from split_manifest.csv.
# Run this once before feature extraction.
#
# Usage: bash bsh/run_generate_pairs.sh

set -euo pipefail
mkdir -p ~/logs
cd "$(dirname "$0")/.."
PYTHONPATH=. uv run python experiments/generate_mixture_pairs.py \
    2>&1 | tee ~/logs/generate_pairs_$(date +%Y%m%d_%H%M%S).log
