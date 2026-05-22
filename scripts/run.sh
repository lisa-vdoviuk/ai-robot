#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
exec python app.py --config "${VOICEPI_CONFIG:-config.yaml}"
