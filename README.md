# VoicePi LLM Assistant v2

A local voice-first AI companion for Raspberry Pi 5 8GB.

This v2 build is English-first because English STT/TTS quality is much better on small offline models. It also changes the conversation architecture so Vosk partial/final segments no longer become separate user turns. A turn is committed only after the browser VAD detects the end of the user's utterance.

## What changed in v2

- English-only assistant, UI, logs, prompts, STT, and TTS.
- Male English Piper voice: `en_US-ryan-high` by default.
- Better default STT model: `vosk-model-en-us-0.22-lgraph`.
- Better default LLM quality profile: `Qwen2.5-3B-Instruct-GGUF` Q4_K_M.
- Client-side VAD now sends audio only during detected human speech, not continuously.
- Barge-in has a higher threshold while TTS is playing to avoid the assistant interrupting itself through speaker echo.
- Server commits a user message only on `speech_end`, not on each Vosk `AcceptWaveform()` segment.
- Visible rationale is always shown: model-generated if tags are followed, safe fallback if not.
- Short/garbled one-word transcripts can trigger a clarification instead of a wrong answer.

> The Explainable AI panel does **not** expose hidden chain-of-thought. It shows a short safe visible rationale, raw model output, prompt/messages, request metrics, and logs. This is suitable for debugging and research instrumentation without trying to extract private reasoning traces.

## 1. Raspberry Pi setup

Recommended:

- Raspberry Pi 5, 8GB RAM.
- Raspberry Pi OS 64-bit.
- Active cooling.
- USB/NVMe SSD or a fast microSD.
- 2-4GB swap if you run other services.

```bash
unzip voicepi-llm-assistant-v2.zip
cd voicepi-llm-assistant-v2
chmod +x scripts/install_pi.sh
./scripts/install_pi.sh
```

The installer creates `.venv`, installs Python dependencies, and builds `llama-cpp-python` with OpenBLAS.

## 2. Download models

Default quality profile:

```bash
source .venv/bin/activate
python scripts/download_models.py
```

Downloads:

- `Qwen/Qwen2.5-3B-Instruct-GGUF/qwen2.5-3b-instruct-q4_k_m.gguf`
- `vosk-model-en-us-0.22-lgraph`
- `rhasspy/piper-voices/en_US-ryan-high`

Faster profile if the Pi gets too slow or hot:

```bash
python scripts/download_models.py --llm fast --stt small --tts medium
```

Then edit `config.yaml` paths:

```yaml
llm:
  model_path: "models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf"
stt:
  model_path: "models/vosk/vosk-model-small-en-us-0.15"
tts:
  model_path: "models/piper/en_US-ryan-medium.onnx"
  config_path: "models/piper/en_US-ryan-medium.onnx.json"
```

## 3. Run

```bash
cp config.example.yaml config.yaml
python app.py --config config.yaml
```

Open on your laptop:

```text
https://<RASPBERRY_PI_IP>:5443
```

Accept the self-signed certificate. Browser microphone access from another device normally requires HTTPS.

## 4. Conversation flow

1. Browser captures the laptop microphone with echo cancellation, noise suppression, and AGC.
2. JavaScript performs lightweight RMS VAD.
3. Audio is sent to the server only during detected human speech.
4. The browser sends `speech_start` and `speech_end` events.
5. Vosk streams partial transcripts, but the server starts the LLM only after `speech_end`.
6. The LLM is instructed to emit:
   - `<rationale>...</rationale>` for the XAI panel.
   - `<answer>...</answer>` for spoken output.
7. Piper synthesizes sentence chunks into WAV files.
8. The browser queues and plays those WAV files.
9. If you talk over the assistant, the browser stops playback and the server cancels the active generation/TTS turn.

## 5. Tuning for quality

### STT

If words are missed, first tune the VAD in the UI:

- Lower `Voice threshold` if your mic is quiet.
- Raise it if room noise creates fake speech.
- Raise `Barge-in threshold` if the assistant still interrupts itself while speaking.
- Use headphones during debugging; full-duplex voice over laptop speakers is hard because speaker echo can look like speech.

For better STT accuracy, keep the default `vosk-model-en-us-0.22-lgraph`. For faster startup and less memory, use `vosk-model-small-en-us-0.15`.

### LLM

Default v2 uses the 3B model because it is noticeably more coherent than 1.5B while still fitting Raspberry Pi 5 8GB in Q4. If latency is too high:

```yaml
llm:
  model_path: "models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf"
  max_tokens: 140
  n_ctx: 1536
```

### TTS

Default voice is `en_US-ryan-high`, a male English Piper voice. If synthesis is too slow:

```yaml
tts:
  model_path: "models/piper/en_US-ryan-medium.onnx"
  config_path: "models/piper/en_US-ryan-medium.onnx.json"
```

## 6. Logs and Explainable AI

The UI shows:

- STT partial and final transcript.
- Raw LLM stream.
- Visible rationale.
- Prompt/messages.
- Generation metrics.
- TTS latency.
- Cancel/barge-in events.

JSONL log file:

```text
logs/voicepi.jsonl
```

## 7. Systemd service

```bash
sudo cp systemd/voicepi-assistant.service.example /etc/systemd/system/voicepi-assistant.service
sudo systemctl daemon-reload
sudo systemctl enable --now voicepi-assistant
```

Edit `WorkingDirectory`, `ExecStart`, and `User` first.

## 8. Smoke test without microphone

Open the page, type a message in the text field, and press Send. This tests LLM, TTS, WebSocket, and UI without STT.

Try:

```text
How are you?
What is two plus two?
If I say something unclear, ask me to repeat it.
```

## 9. Known limits

- A 3B local model is much smarter than tiny models, but it is still not a cloud-scale assistant.
- Piper is fast and local, but not ElevenLabs/XTTS quality.
- Full-duplex speech with laptop speakers depends heavily on browser echo cancellation and room acoustics.
- Self-signed HTTPS is fine for LAN testing, but use a proper certificate or reverse proxy for a long-term setup.

---

## v3 robot-control layer: Raspberry Pi LLM server → ESP32 → motors

This build adds a safe tool-calling layer for a small wheeled robot.

The flow is:

1. Browser microphone captures speech.
2. Raspberry Pi runs STT and sends the transcript to the local LLM.
3. Before the spoken answer is generated, the same local LLM runs a short constrained planner.
4. The planner may output one validated robot command:
   - `none`
   - `stop`
   - `move forward`
   - `move backward`
   - `turn left`
   - `turn right`
5. The Python server clamps duration and speed, then calls the ESP32 HTTP API.
6. The final assistant answer includes the robot tool result, so the assistant can say things like “Moving forward.” or “I couldn’t reach the robot.”

This is intentionally not a keyword router. The LLM decides whether the user is asking for physical movement, while the server validates the decision before anything reaches the motors.

### Important fix: TTS no longer speaks `</answer>`

v3 adds a streaming `<answer>` parser. It emits only text inside the answer tag and stops before partial closing tags such as `</ans`, `</answer`, or `</answer>` can enter the TTS queue.

### Python configuration

Copy your existing `config.yaml`, then add or edit this block:

```yaml
robot:
  enabled: true
  base_url: "http://ESP32_IP_FROM_SERIAL_MONITOR"
  token: "change-me-robot-token"
  timeout_s: 2.0

  min_confidence: 0.55
  min_speed: 0.20
  max_speed: 0.75
  default_speed: 0.45
  min_duration_ms: 150
  max_duration_ms: 1800
  default_duration_ms: 700
  planner_max_tokens: 120

  stop_on_barge_in: true
  stop_on_disconnect: true
```

Keep `robot.enabled: false` until the ESP32 is flashed and the wheels are lifted off the table for the first test.

### ESP32 firmware

Firmware is in:

```text
esp32/voicepi_motor_controller.ino
```

Edit these values before flashing:

```cpp
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* API_TOKEN = "change-me-robot-token";
```

Also verify the motor pins:

```cpp
const int LEFT_IN1 = 26;
const int LEFT_IN2 = 27;
const int LEFT_PWM = 25;

const int RIGHT_IN1 = 32;
const int RIGHT_IN2 = 33;
const int RIGHT_PWM = 14;
```

The sketch assumes a two-channel H-bridge style layout: left-side motors on one channel and right-side motors on one channel. If your module exposes four independent motor channels, either wire/bridge them as left/right pairs or adapt `setSide()` and `drive()`.

The PlatformIO file is included:

```bash
cd esp32
pio run -t upload
pio device monitor
```

The sketch uses ArduinoJson. If you use Arduino IDE instead of PlatformIO, install ArduinoJson from Library Manager.

### Direct ESP32 test from the Raspberry Pi

After flashing, the ESP32 prints its IP address in Serial Monitor. Test it before enabling the LLM robot tool:

```bash
export ROBOT_IP="192.168.1.50"
export ROBOT_TOKEN="change-me-robot-token"

curl -H "X-VoicePi-Token: $ROBOT_TOKEN" "http://$ROBOT_IP/api/status"

curl -X POST "http://$ROBOT_IP/api/move" \
  -H "Content-Type: application/json" \
  -H "X-VoicePi-Token: $ROBOT_TOKEN" \
  -d '{"direction":"forward","speed":0.35,"duration_ms":400}'

curl -X POST "http://$ROBOT_IP/api/stop" \
  -H "X-VoicePi-Token: $ROBOT_TOKEN"
```

For first tests, lift the robot so the wheels are not touching the table. If left/right are reversed, swap motor wires or set `INVERT_LEFT` / `INVERT_RIGHT` in the sketch.

### Voice examples

Try short commands first:

- “Move forward a little.”
- “Go back.”
- “Turn left.”
- “Turn right for a moment.”
- “Stop.”

Normal conversation should produce `action: none` in the Robot Tools panel.

### Debugging

The web UI now has a `robot tools` tab under Explainable AI / Trace. It shows:

- raw LLM planner output,
- validated command,
- ESP32 HTTP result,
- errors or skipped actions.

The same data is written into `logs/voicepi.jsonl` with source `robot`.

---

## v4 camera + scene-awareness vision layer

This build adds a Raspberry Pi camera service without removing the existing STT, TTS, LLM chat, memory or ESP32 robot-control layer.

### What is included

- Picamera2 integration inside the Flask app.
- Live MJPEG stream in the web control panel.
- `/camera/stream.mjpg` for the live stream.
- `/camera/snapshot.jpg` for the latest frame.
- `/camera/status` for camera + vision status.
- A Camera / Vision panel with live camera, current observation and manual “Analyze current frame”.
- Local scene awareness in `voicepi/vision.py`: object detection, rough left/center/right zones, motion detection and possible close-obstacle hints.
- Camera observations are attached to the LLM prompt as `[CAMERA OBSERVATION]`, so the robot can answer questions like “What do you see?” or “Is there something moving?”

### Important architecture note

The bundled local Qwen GGUF model is a text-only LLM. It cannot directly see pixels. The `voicepi/vision.py` service converts the camera frame into concise structured text first, such as:

```text
summary=Objects: person in center 0.82, bottle in left 0.66. Motion detected in the right zone.
scene={'person_count': 1, 'close_obstacles': [], 'object_zones': {'center': ['person'], 'left': ['bottle']}, 'attention': 'person_visible'}
objects=[{'label': 'person', 'confidence': 0.82, 'zone': 'center', 'area_ratio': 0.18}]
motion={'detected': True, 'zone': 'right', 'changed_area_ratio': 0.024}
confidence=0.82
```

The LLM then reasons over that observation. This keeps the system light enough for a Raspberry Pi 5 and gives you a strong base for later navigation, person-aware interaction and safety behavior.

### Camera dependencies on Raspberry Pi OS

Picamera2 is normally installed through apt, not pip. The installer installs the apt camera/OpenCV packages and creates the venv with `--system-site-packages` so the app can import them:

```bash
sudo apt-get install -y python3-picamera2 python3-opencv python3-numpy
python3 -m venv --system-site-packages .venv
```

If you already have an old `.venv` that cannot import `picamera2` or `cv2`, recreate it:

```bash
rm -rf .venv
./scripts/install_pi.sh
```

### Config block

Add this to your existing `config.yaml`, or copy the new `config.example.yaml`:

```yaml
camera:
  enabled: true
  width: 640
  height: 480

vision:
  enabled: true
  poll_interval_s: 2.0
  snapshot_on_turn: true
  always_attach_to_prompt: true
  max_prompt_age_s: 3.0
  log_poll_observations: false
  obstacle_area_ratio: 0.10
  motion:
    enabled: true
    threshold: 24
    min_area_ratio: 0.01
  object_detection:
    enabled: true
    model_path: "models/vision/mobilenet_iter_73000.caffemodel"
    config_path: "models/vision/deploy.prototxt"
    confidence_threshold: 0.45
    max_objects: 6
```

Download the small object detector files:

```bash
python scripts/download_models.py --skip-llm --skip-stt --skip-tts --vision mobilenet-ssd
```

### Test prompts

Type or say:

```text
What does the camera see?
Is there a person in front of you?
Do you see any close obstacle?
Where is the moving object?
```

For the first tests, keep the robot wheels lifted off the table and keep `robot.enabled: false` until you trust the camera and ESP32 path.
