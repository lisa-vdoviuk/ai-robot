#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import quote

import requests
from tqdm.auto import tqdm

# Avoid Hugging Face Xet/CAS issues on Raspberry Pi and restricted networks.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_HF_TRANSFER", "1")

ROOT = Path(__file__).resolve().parents[1]

LLM_PROFILES = {
    "quality": (
        "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "qwen2.5-3b-instruct-q4_k_m.gguf",
        "~2.0 GB",
    ),
    "fast": (
        "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "~1.0 GB",
    ),
    "tiny": (
        "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "~0.4 GB",
    ),
}

PIPER_REPO = "rhasspy/piper-voices"
PIPER_PROFILES = {
    "high": (
        "en/en_US/amy/high/en_US-amy-high.onnx",
        "en/en_US/amy/high/en_US-amy-high.onnx.json",
    ),
    "medium": (
        "en/en_US/ryan/medium/en_US-ryan-medium.onnx",
        "en/en_US/ryan/medium/en_US-ryan-medium.onnx.json",
    ),
    "ryan-high": (
        "en/en_US/ryan/high/en_US-ryan-high.onnx",
        "en/en_US/ryan/high/en_US-ryan-high.onnx.json",
    ),
    "ryan-medium": (
        "en/en_US/ryan/medium/en_US-ryan-medium.onnx",
        "en/en_US/ryan/medium/en_US-ryan-medium.onnx.json",
    ),
    # Pleasant female English voice option. Change tts.model_path in config.yaml after downloading.
    "amy-high": (
        "en/en_US/amy/high/en_US-amy-high.onnx",
        "en/en_US/amy/high/en_US-amy-high.onnx.json",
    ),
    "amy-medium": (
        "en/en_US/amy/medium/en_US-amy-medium.onnx",
        "en/en_US/amy/medium/en_US-amy-medium.onnx.json",
    ),
        "lessac-medium": (
        "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
    ),
    "lessac-high": (
        "en/en_US/lessac/high/en_US-lessac-high.onnx",
        "en/en_US/lessac/high/en_US-lessac-high.onnx.json",
    ),
    "ukrainian-medium": (
        "uk/uk_UA/ukrainian_tts/medium/uk_UA-ukrainian_tts-medium.onnx",
        "uk/uk_UA/ukrainian_tts/medium/uk_UA-ukrainian_tts-medium.onnx.json",
    ),
}

VISION_PROFILES = {
    "mobilenet-ssd": (
        ("https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt", "models/vision/deploy.prototxt"),
        ("https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel", "models/vision/mobilenet_iter_73000.caffemodel"),
    ),
}

VOSK_PROFILES = {
    "quality": "vosk-model-en-us-0.22-lgraph",
    "small": "vosk-model-small-en-us-0.15",
}


def human_bytes(n: int | float) -> str:
    value = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def hf_resolve_url(repo: str, filename: str) -> str:
    # Quote each path component but keep slashes in nested filenames.
    quoted = "/".join(quote(part) for part in filename.split("/"))
    return f"https://huggingface.co/{repo}/resolve/main/{quoted}?download=true"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def download_url(url: str, dst: Path, label: str, retries: int = 5) -> None:
    ensure_parent(dst)
    part = dst.with_suffix(dst.suffix + ".part")

    if dst.exists() and dst.stat().st_size > 0:
        print(f"✓ {dst.relative_to(ROOT)} already exists ({human_bytes(dst.stat().st_size)})", flush=True)
        return

    for attempt in range(1, retries + 1):
        resume_from = part.stat().st_size if part.exists() else 0
        headers = {
            "User-Agent": "voicepi-model-downloader/2.1",
        }
        if resume_from:
            headers["Range"] = f"bytes={resume_from}-"

        print(f"\nDownloading {label}", flush=True)
        print(f"URL: {url}", flush=True)
        if resume_from:
            print(f"Resuming from {human_bytes(resume_from)}", flush=True)

        try:
            with requests.get(url, stream=True, allow_redirects=True, timeout=(20, 60), headers=headers) as response:
                if response.status_code == 416:
                    # Partial file is probably complete but server refuses range.
                    part.rename(dst)
                    print(f"✓ {dst.relative_to(ROOT)}", flush=True)
                    return

                response.raise_for_status()

                # If server ignored Range, restart partial file from 0.
                mode = "ab"
                if resume_from and response.status_code == 200:
                    print("Server did not resume; restarting this file.", flush=True)
                    resume_from = 0
                    mode = "wb"
                elif not resume_from:
                    mode = "wb"

                content_length = int(response.headers.get("Content-Length") or 0)
                total = content_length + resume_from if content_length else None
                chunk_size = 1024 * 1024

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
                    last_write = time.time()
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        bar.update(len(chunk))
                        last_write = time.time()

                    if time.time() - last_write > 60:
                        print("No data was received for a long time.", flush=True)

            part.rename(dst)
            print(f"✓ {dst.relative_to(ROOT)} ({human_bytes(dst.stat().st_size)})", flush=True)
            return
        except KeyboardInterrupt:
            print("\nInterrupted. Partial file kept for resume:", part, flush=True)
            raise
        except Exception as exc:
            print(f"Attempt {attempt}/{retries} failed: {exc}", flush=True)
            if attempt == retries:
                print("\nDownload failed. You can rerun this command; .part files will be resumed when possible.", flush=True)
                raise
            time.sleep(min(2 * attempt, 10))


def download_llm(profile: str) -> None:
    repo, filename, size_hint = LLM_PROFILES[profile]
    dst = ROOT / "models" / "llm" / filename
    url = hf_resolve_url(repo, filename)
    download_url(url, dst, f"LLM {profile}: {repo}/{filename} ({size_hint})")


def download_piper(profile: str) -> None:
    for filename in PIPER_PROFILES[profile]:
        dst = ROOT / "models" / "piper" / Path(filename).name
        url = hf_resolve_url(PIPER_REPO, filename)
        download_url(url, dst, f"Piper voice {profile}: {filename}")


def download_vision(profile: str) -> None:
    for url, rel_path in VISION_PROFILES[profile]:
        dst = ROOT / rel_path
        download_url(url, dst, f"Vision object detector {profile}: {rel_path}")


def download_vosk(profile: str) -> None:
    model_name = VOSK_PROFILES[profile]
    dst_dir = ROOT / "models" / "vosk" / model_name
    if dst_dir.exists():
        print(f"✓ {dst_dir.relative_to(ROOT)} already exists", flush=True)
        return

    zip_path = ROOT / "models" / "vosk" / f"{model_name}.zip"
    url = f"https://alphacephei.com/vosk/models/{model_name}.zip"
    download_url(url, zip_path, f"Vosk English STT model {profile}: {model_name}")

    print("Extracting Vosk model...", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(zip_path.parent)
    zip_path.unlink(missing_ok=True)
    print(f"✓ {dst_dir.relative_to(ROOT)}", flush=True)


def print_disk_hint() -> None:
    usage = shutil.disk_usage(ROOT)
    print(
        f"Disk free in project folder: {human_bytes(usage.free)}. "
        "Quality profile needs roughly 4-5 GB free during downloads.",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--skip-stt", action="store_true")
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("--skip-vision", action="store_true")
    parser.add_argument("--llm", choices=LLM_PROFILES, default="quality")
    parser.add_argument("--stt", choices=VOSK_PROFILES, default="quality")
    parser.add_argument("--tts", choices=PIPER_PROFILES, default="amy-medium")
    parser.add_argument("--vision", choices=VISION_PROFILES, default="mobilenet-ssd")
    parser.add_argument(
        "--small-first",
        action="store_true",
        help="Download STT and TTS before the large LLM, useful for testing network/progress quickly.",
    )
    args = parser.parse_args()

    print_disk_hint()

    tasks = []
    if not args.skip_llm:
        tasks.append(("llm", lambda: download_llm(args.llm)))
    if not args.skip_stt:
        tasks.append(("stt", lambda: download_vosk(args.stt)))
    if not args.skip_tts:
        tasks.append(("tts", lambda: download_piper(args.tts)))
    if not args.skip_vision:
        tasks.append(("vision", lambda: download_vision(args.vision)))

    if args.small_first:
        order = {"stt": 0, "tts": 1, "vision": 2, "llm": 3}
        tasks.sort(key=lambda item: order[item[0]])

    for _, task in tasks:
        task()

    print("\nDone. Check config.yaml paths if you changed profiles.", flush=True)
    if args.llm == "fast":
        print("Fast LLM path: models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf", flush=True)
    if args.llm == "tiny":
        print("Tiny LLM path: models/llm/qwen2.5-0.5b-instruct-q4_k_m.gguf", flush=True)
    if args.stt == "small":
        print("Small STT path: models/vosk/vosk-model-small-en-us-0.15", flush=True)
    if args.tts == "ryan-medium":
        print("Medium TTS path: models/piper/en_US-ryan-medium.onnx", flush=True)
    if args.tts == "ryan-high":
        print("Ryan high TTS path: models/piper/en_US-ryan-high.onnx", flush=True)
    if args.tts == "amy-high":
        print("Amy high TTS path: models/piper/en_US-amy-high.onnx", flush=True)
    if not args.skip_vision:
        print("Vision object model paths: models/vision/deploy.prototxt and models/vision/mobilenet_iter_73000.caffemodel", flush=True)


if __name__ == "__main__":
    main()
