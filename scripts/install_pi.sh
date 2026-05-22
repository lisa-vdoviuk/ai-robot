#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "$(uname -m)" != "aarch64" && "$(uname -m)" != "arm64" ]]; then
  echo "Warning: this script is tuned for Raspberry Pi OS 64-bit / ARM64, current arch: $(uname -m)"
fi

sudo apt-get update
sudo apt-get install -y \
  python3-venv python3-dev build-essential cmake pkg-config \
  libopenblas-dev libssl-dev espeak-ng ffmpeg unzip wget git \
  python3-picamera2 python3-opencv python3-numpy

# Picamera2 is normally installed through apt on Raspberry Pi OS.
# --system-site-packages lets the venv import python3-picamera2/python3-opencv.
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# Build llama-cpp-python against OpenBLAS on Raspberry Pi CPU.
export CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
export FORCE_CMAKE=1
python -m pip install --no-cache-dir --force-reinstall llama-cpp-python

# Install the rest. The llama-cpp-python line in requirements will be satisfied.
python -m pip install --no-cache-dir -r requirements.txt

if [[ ! -f config.yaml ]]; then
  cp config.example.yaml config.yaml
  echo "Created config.yaml"
fi

echo
cat <<'MSG'
Install complete.

Next:
  source .venv/bin/activate
  python scripts/download_models.py
  python app.py --config config.yaml

Open from your laptop:
  https://<raspberry-pi-ip>:5443

Accept the self-signed certificate so the browser can use the laptop microphone.
MSG
