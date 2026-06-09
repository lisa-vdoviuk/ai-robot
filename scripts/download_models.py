#!/usr/bin/env python3
"""Download only the models used by the current robot build.

Current runtime stack:
  - STT: Vosk English model
  - TTS: Kokoro ONNX int8 + voices
  - Vision: YOLO11n ONNX

Groq is the default LLM provider, so this script intentionally does not pull a
large local GGUF model unless you add that path back yourself.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import requests
from tqdm.auto import tqdm

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_HF_TRANSFER", "1")

ROOT = Path(__file__).resolve().parents[1]

VOSK_PROFILES = {
    "quality": "vosk-model-en-us-0.22-lgraph",
    "small": "vosk-model-small-en-us-0.15",
}

KOKORO_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
KOKORO_FILES = [
    (f"{KOKORO_RELEASE}/kokoro-v1.0.int8.onnx", "models/kokoro/kokoro-v1.0.int8.onnx"),
    (f"{KOKORO_RELEASE}/voices-v1.0.bin", "models/kokoro/voices-v1.0.bin"),
]

VISION_FILES = [
    (
        "https://huggingface.co/Ultralytics/assets/resolve/main/yolo11n.onnx?download=true",
        "models/vision/yolo11n.onnx",
    ),
]


def human_bytes(n: int | float) -> str:
    value = float(n)
    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def download_url(url: str, dst: Path, label: str, retries: int = 5) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")

    if dst.exists() and dst.stat().st_size > 0:
        print(f"✓ {dst.relative_to(ROOT)} already exists ({human_bytes(dst.stat().st_size)})", flush=True)
        return

    for attempt in range(1, retries + 1):
        resume_from = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "voicepi-model-downloader/3.0"}
        if resume_from:
            headers["Range"] = f"bytes={resume_from}-"

        print(f"\nDownloading {label}", flush=True)
        print(f"URL: {url}", flush=True)
        if resume_from:
            print(f"Resuming from {human_bytes(resume_from)}", flush=True)

        try:
            with requests.get(url, stream=True, allow_redirects=True, timeout=(20, 60), headers=headers) as response:
                if response.status_code == 416:
                    part.rename(dst)
                    print(f"✓ {dst.relative_to(ROOT)}", flush=True)
                    return
                response.raise_for_status()

                mode = "ab" if resume_from else "wb"
                if resume_from and response.status_code == 200:
                    print("Server did not resume; restarting this file.", flush=True)
                    resume_from = 0
                    mode = "wb"

                content_length = int(response.headers.get("Content-Length") or 0)
                total = content_length + resume_from if content_length else None
                with open(part, mode) as fh, tqdm(
                    total=total,
                    initial=resume_from,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    mininterval=0.5,
                    desc=dst.name,
                    file=sys.stdout,
                ) as bar:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            fh.write(chunk)
                            bar.update(len(chunk))
            part.rename(dst)
            print(f"✓ {dst.relative_to(ROOT)} ({human_bytes(dst.stat().st_size)})", flush=True)
            return
        except KeyboardInterrupt:
            print(f"\nInterrupted. Partial file kept for resume: {part}", flush=True)
            raise
        except Exception as exc:
            print(f"Attempt {attempt}/{retries} failed: {exc}", flush=True)
            if attempt == retries:
                raise
            time.sleep(min(2 * attempt, 10))


def download_vosk(profile: str) -> None:
    model_name = VOSK_PROFILES[profile]
    dst_dir = ROOT / "models" / "vosk" / model_name
    if dst_dir.exists():
        print(f"✓ {dst_dir.relative_to(ROOT)} already exists", flush=True)
        return

    zip_path = ROOT / "models" / "vosk" / f"{model_name}.zip"
    download_url(f"https://alphacephei.com/vosk/models/{model_name}.zip", zip_path, f"Vosk STT {profile}: {model_name}")
    print("Extracting Vosk model...", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(zip_path.parent)
    zip_path.unlink(missing_ok=True)
    print(f"✓ {dst_dir.relative_to(ROOT)}", flush=True)


def download_kokoro() -> None:
    for url, rel_path in KOKORO_FILES:
        download_url(url, ROOT / rel_path, f"Kokoro TTS: {Path(rel_path).name}")


def download_vision() -> None:
    for url, rel_path in VISION_FILES:
        download_url(url, ROOT / rel_path, f"YOLO object detector: {Path(rel_path).name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download VoicePi robot models")
    parser.add_argument("--skip-stt", action="store_true")
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("--skip-vision", action="store_true")
    parser.add_argument("--stt", choices=VOSK_PROFILES, default="quality")
    parser.add_argument("--small-first", action="store_true", help="Download the smaller Kokoro/YOLO files before STT")
    # Backward-compatible no-op flags used by older README/install commands.
    parser.add_argument("--skip-llm", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--tts-engine", choices=["kokoro"], default="kokoro", help=argparse.SUPPRESS)
    args = parser.parse_args()

    free = shutil.disk_usage(ROOT).free
    print(f"Disk free in project folder: {human_bytes(free)}", flush=True)

    tasks: list[tuple[str, object]] = []
    if not args.skip_stt:
        tasks.append(("stt", lambda: download_vosk(args.stt)))
    if not args.skip_tts:
        tasks.append(("tts", download_kokoro))
    if not args.skip_vision:
        tasks.append(("vision", download_vision))

    if args.small_first:
        order = {"tts": 0, "vision": 1, "stt": 2}
        tasks.sort(key=lambda item: order[item[0]])

    for _, task in tasks:
        task()  # type: ignore[misc]

    print("\nDone. Expected paths:", flush=True)
    print("  models/vosk/<selected-vosk-model>", flush=True)
    print("  models/kokoro/kokoro-v1.0.int8.onnx", flush=True)
    print("  models/kokoro/voices-v1.0.bin", flush=True)
    print("  models/vision/yolo11n.onnx", flush=True)


if __name__ == "__main__":
    main()
