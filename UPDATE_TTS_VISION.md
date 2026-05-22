# Update: smoother TTS + basic object recognition

This build includes the previous camera/vision hotfix plus:

- safer TTS text normalization so Piper does not speak XML tags, markdown, emoji, URLs, or punctuation such as isolated `!`;
- TTS chunk coalescing to reduce short WAV gaps and make speech feel less choppy;
- optional Piper controls: `noise_scale`, `noise_w`, `sentence_silence`, `normalize_text`, `coalesce_wait_s`, `chunk_max_chars`;
- default voice changed in `config.example.yaml` to `en_US-amy-high`;
- OpenCV DNN + MobileNet-SSD object recognition for basic classes such as `person`, `bottle`, `chair`, `dog`, `cat`, `car`;
- object detection output added to camera/vision UI logs and LLM prompt context.

## Install/update on Raspberry Pi

From the project root:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_models.py --skip-llm --skip-stt --tts amy-high --vision mobilenet-ssd
```

If your `.venv` was created without system packages, recreate it so Picamera2/OpenCV from apt are visible:

```bash
rm -rf .venv
./scripts/install_pi.sh
python scripts/download_models.py --skip-llm --skip-stt --tts amy-high --vision mobilenet-ssd
```

If you already have `config.yaml`, copy the updated `tts:` and `vision:` sections from `config.example.yaml` into your active `config.yaml`.

Restart:

```bash
./scripts/run.sh
# or
sudo systemctl restart voicepi-assistant
```
