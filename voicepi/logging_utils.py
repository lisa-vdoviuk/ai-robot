from __future__ import annotations

import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any


class JsonlLogger:
    def __init__(self, path: Path, console: bool = True) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.console = console
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    def event(self, source: str, level: str, message: str, **fields: Any) -> dict[str, Any]:
        record = {
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source": source,
            "level": level,
            "message": message,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False)
        with self.lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        if self.console:
            getattr(logging, level.lower(), logging.info)("[%s] %s", source, message)
        return record
