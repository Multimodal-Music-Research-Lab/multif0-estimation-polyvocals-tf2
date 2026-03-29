#!/usr/bin/env bash
# setup.sh — Dev environment setup for multif0-estimation-polyvocals-tf2
# Uses uv for Python package management.
# Requirements: Debian/Ubuntu Linux, NVIDIA GPU + drivers (CUDA toolkit is bundled via pip)
set -euo pipefail

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
echo ">>> Installing system packages..."
sudo apt-get update -y
sudo apt-get install -y sox ffmpeg rubberband-cli

# ---------------------------------------------------------------------------
# 2. Install uv (if not already present)
# ---------------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
    echo ">>> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # The installer places uv in ~/.local/bin
    export PATH="$HOME/.local/bin:$PATH"
fi
echo ">>> uv version: $(uv --version)"

# ---------------------------------------------------------------------------
# 3. Install Python 3.9
# ---------------------------------------------------------------------------
echo ">>> Installing Python 3.9 via uv..."
uv python install 3.9

# ---------------------------------------------------------------------------
# 4. Create virtual environment
# ---------------------------------------------------------------------------
echo ">>> Creating .venv with Python 3.9..."
uv venv --python 3.9 .venv
# shellcheck disable=SC1091
source .venv/bin/activate

# ---------------------------------------------------------------------------
# 5. Install Python packages (versions pinned from environment.yml)
# ---------------------------------------------------------------------------
echo ">>> Installing Python packages..."
uv pip install \
    "tensorflow[and-cuda]==2.17.0" \
    "keras==3.6.0" \
    "numpy==1.26.4" \
    "scipy==1.13.1" \
    "pandas==2.2.3" \
    "scikit-learn==1.5.2" \
    "matplotlib==3.9.2" \
    "h5py==3.12.1" \
    "tensorboard==2.17.1" \
    "numba==0.60.0" \
    "llvmlite==0.43.0" \
    "librosa==0.10.2.post1" \
    "soundfile" \
    "audioread==3.0.1" \
    "soxr==0.3.7" \
    "pydub==0.25.1" \
    "mir-eval==0.7" \
    "jams==0.3.4" \
    "pumpp==0.6.0" \
    "muda==0.4.1" \
    "pyrubberband==0.4.0" \
    "pescador==3.0.0"

# ---------------------------------------------------------------------------
# 6. Post-install reminders
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Setup complete. Before running the pipeline:"
echo ""
echo " 1. Set the required environment variable:"
echo "      export DAST_BASE=/path/to/dAST"
echo ""
echo " 2. Copy the impulse response file (not tracked in git):"
echo "      cp /path/to/IR_greathall.wav ir/IR_greathall.wav"
echo ""
echo " 3. Activate the environment in future sessions:"
echo "      source .venv/bin/activate"
echo ""
echo " 4. Verify GPU visibility:"
echo "      python -c \"import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))\""
echo "============================================================"
