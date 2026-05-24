#!/usr/bin/env bash
# Raspberry Pi OS 64-bit (aarch64) setup script for VoicePi.
# Run once after cloning the repo.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "$(uname -m)" != "aarch64" && "$(uname -m)" != "arm64" ]]; then
  echo "Warning: tuned for Raspberry Pi OS 64-bit / ARM64, current arch: $(uname -m)"
fi

echo "==> Installing system packages..."
sudo apt-get update
sudo apt-get install -y \
  python3-venv python3-dev build-essential cmake pkg-config \
  libopenblas-dev libssl-dev espeak-ng ffmpeg unzip wget git \
  python3-picamera2 python3-opencv python3-numpy \
  portaudio19-dev libsndfile1

# --system-site-packages lets the venv import apt-installed
# python3-picamera2 and python3-opencv (ARM-optimised builds).
echo "==> Creating virtual environment..."
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# onnxruntime -- ARM64 wheels available on PyPI since 1.17.
echo "==> Installing onnxruntime..."
pip install --no-cache-dir "onnxruntime>=1.17.0"

# kokoro-onnx -- no PyTorch, ONNX Runtime based, works on Pi.
echo "==> Installing kokoro-onnx..."
pip install --no-cache-dir "kokoro-onnx>=0.4.0"

# edge-tts -- Microsoft Edge neural TTS, free, internet required.
echo "==> Installing edge-tts..."
pip install --no-cache-dir "edge-tts>=6.1.9"

# LLM (local engine only -- skip if you use groq/openai in config.yaml).
# Groq is the default; uncomment the block below only if llm.engine = "llama_cpp".
# export CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
# export FORCE_CMAKE=1
# pip install --no-cache-dir --force-reinstall llama-cpp-python

echo "==> Installing remaining Python dependencies..."
pip install --no-cache-dir -r requirements.txt

if [[ ! -f config.yaml ]]; then
  cp config.example.yaml config.yaml 2>/dev/null || true
  echo "Created config.yaml from example"
fi

echo ""
cat <<'MSG'
=== Install complete ===

Next steps:
  1. source .venv/bin/activate

  2. Download models:
       # For offline Kokoro TTS + YOLO + STT (no LLM -- uses Groq API):
       python scripts/download_models.py --skip-llm

       # Or with local LLM:
       python scripts/download_models.py

  3. Edit config.yaml:
       tts.engine = "edge"    # best quality, needs internet (Microsoft Edge TTS)
       tts.engine = "kokoro"  # best offline quality

  4. Run:
       python app.py --config config.yaml

  5. Open from your laptop:
       https://<raspberry-pi-ip>:5443
       (accept the self-signed certificate so browser can use the mic)

Camera troubleshooting:
  - CSI camera (Pi ribbon cable): enable with  sudo raspi-config  -> Interface -> Camera
  - USB camera: works automatically via cv2 fallback (/dev/video0)
  - Check camera status at: https://<pi-ip>:5443/camera/status
MSG
