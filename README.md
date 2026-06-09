# VoicePi Robot Assistant

A focused Raspberry Pi 5 robot assistant with:

- browser microphone + WebSocket audio
- Vosk speech recognition
- Groq LLM chat/planning
- Kokoro ONNX offline TTS
- Pi camera stream
- YOLO11n ONNX object detection
- frame-difference motion detection
- optional ESP32 motor controller API

The repository is intentionally kept small. Model binaries, logs, local configs,
old patches, and export artifacts are ignored.

## Repository layout

```text
app.py                         Flask/WebSocket entry point
config.yaml                    local runtime config
config.example.yaml            copy for a fresh setup
voicepi/
  camera.py                    Picamera2 + OpenCV fallback camera manager
  vision.py                    YOLO11n ONNX + motion scene analyzer
  tts_kokoro.py                Kokoro ONNX TTS engine
  stt_vosk.py                  Vosk streaming STT
  llm_engine.py                Groq/local LLM wrapper
  session.py                   one browser conversation session
  robot_controller.py          ESP32 HTTP robot client
  decision_manager.py          safety filter for robot commands
scripts/
  install_pi.sh                Raspberry Pi setup
  download_models.py           downloads STT/TTS/YOLO models
  run.sh                       starts the app from the venv
esp32/                         optional motor controller firmware
```

## Raspberry Pi 5 setup

```bash
git clone <your-repo-url> ai-robot
cd ai-robot
bash scripts/install_pi.sh
source .venv/bin/activate
export GROQ_API_KEY="your_key_here"
python scripts/download_models.py --skip-llm --small-first
python app.py --config config.yaml
```

Open the UI from your laptop:

```text
https://<raspberry-pi-ip>:5443
```

Accept the self-signed LAN certificate so the browser can use the microphone.

## Required model paths

```text
models/vosk/vosk-model-en-us-0.22-lgraph/
models/kokoro/kokoro-v1.0.int8.onnx
models/kokoro/voices-v1.0.bin
models/vision/yolo11n.onnx
```

The `.pt` YOLO file is not needed for runtime. Use it only outside the repo for
training/export experiments.

## YOLO object recognition notes

The included detector expects a COCO-trained YOLO11n model. Good test objects:

```text
person, cup, bottle, chair, laptop, keyboard, mouse, cell phone, book, scissors
```

Objects such as `pen`, `hand`, and `palm` are not standard COCO classes. For
those, add a separate hand detector or fine-tune a custom YOLO model later.

Check detector status:

```bash
curl -k https://localhost:5443/camera/status
```

Useful fields:

- `resolved_model_path` should point to `models/vision/yolo11n.onnx`.
- `backend` should become `onnxruntime-yolo` after the first analysis.
- `last_stats.max_confidence_before_filter` shows whether YOLO sees anything before thresholding.

Trigger manual analysis:

```bash
curl -k https://localhost:5443/camera/analyze
```

or use the Analyze button in the UI.

## Kokoro TTS profile

This build uses Kokoro only. The default profile is tuned for Raspberry Pi 5 CPU:

```yaml
tts:
  engine: "kokoro"
  stream_during_generation: true
  sentence_min_chars: 18
  sentence_max_chars: 150
  chunk_max_chars: 190
  coalesce_wait_s: 0.02
  kokoro:
    model_path: "models/kokoro/kokoro-v1.0.int8.onnx"
    voices_path: "models/kokoro/voices-v1.0.bin"
    speed: 1.05
    threads: 2
    warmup: true
```

For better pronunciation of robot terms, edit `tts.pronunciation` in
`config.yaml`. For higher quality but slower speech, test the non-int8 Kokoro
model later; the int8 model is the safer default on Pi 5.

## Camera troubleshooting

- CSI camera: enable camera support in Raspberry Pi OS if needed.
- USB camera: OpenCV fallback uses `/dev/video0` by default.
- Status endpoint: `https://<pi-ip>:5443/camera/status`
- Snapshot endpoint: `https://<pi-ip>:5443/camera/snapshot.jpg`

## Systemd

Edit paths in `systemd/voicepi-assistant.service.example`, then:

```bash
sudo cp systemd/voicepi-assistant.service.example /etc/systemd/system/voicepi-assistant.service
sudo systemctl daemon-reload
sudo systemctl enable --now voicepi-assistant
```

## Development checks

```bash
PYTHONPATH=. pytest -q
python -m compileall app.py voicepi scripts/download_models.py
```
