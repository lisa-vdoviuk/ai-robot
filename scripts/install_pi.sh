#!/usr/bin/env bash
# Raspberry Pi OS 64-bit setup for the current VoicePi robot build.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "$(uname -m)" != "aarch64" && "$(uname -m)" != "arm64" ]]; then
  echo "Warning: this setup is tuned for Raspberry Pi OS 64-bit / ARM64. Current arch: $(uname -m)"
fi

echo "==> Installing system packages"
sudo apt-get update
sudo apt-get install -y \
  python3-venv python3-dev build-essential cmake pkg-config \
  libopenblas-dev libssl-dev espeak-ng ffmpeg unzip wget git \
  python3-picamera2 python3-opencv python3-numpy \
  libsndfile1

echo "==> Creating virtual environment"
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# Keep ONNX/Kokoro from occupying all Pi 5 cores and starving camera/websocket work.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

echo "==> Installing Python dependencies"
pip install --no-cache-dir -r requirements.txt

if [[ ! -f config.yaml ]]; then
  cp config.example.yaml config.yaml
  echo "Created config.yaml from config.example.yaml"
fi

cat <<'MSG'

=== Install complete ===

Next steps:
  1. source .venv/bin/activate
  2. export GROQ_API_KEY="your_key_here"
  3. python scripts/download_models.py --skip-llm --small-first
  4. python app.py --config config.yaml
  5. Open https://<raspberry-pi-ip>:5443 from your laptop and accept the LAN certificate.

Camera troubleshooting:
  - CSI camera: enable camera support in raspi-config if needed.
  - USB camera: uses the OpenCV fallback /dev/video0.
  - Check status at https://<pi-ip>:5443/camera/status
MSG
