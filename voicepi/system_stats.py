"""Raspberry Pi system telemetry: CPU temperature, CPU load, and RAM usage.

Pure standard library (reads /proc and /sys), so there is no extra dependency and it
also runs on a normal Linux dev box (temperature simply reads as null there).

A small background thread keeps a fresh snapshot because CPU utilisation must be
computed from the delta between two /proc/stat reads. The Flask endpoint just returns
the latest cached snapshot, so requests are instant and never block on sampling.

Each metric carries a ``severity`` ("ok" | "warn" | "critical") computed from
configurable thresholds, which the dashboard uses to highlight values.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Optional


def _severity(value: Optional[float], warn: float, critical: float) -> str:
    if value is None:
        return "unknown"
    if value >= critical:
        return "critical"
    if value >= warn:
        return "warn"
    return "ok"


def _read_first_float(path: str) -> Optional[float]:
    try:
        return float(Path(path).read_text().strip())
    except Exception:
        return None


class SystemStatsMonitor:
    def __init__(self, cfg=None, logger=None) -> None:
        self.logger = logger
        get = (cfg.get if cfg is not None else (lambda k, d=None: d))
        self.enabled = bool(get("system.enabled", True))
        self.interval_s = float(get("system.interval_s", 2.0))
        self.temp_warn = float(get("system.temp_warn_c", 70.0))
        self.temp_critical = float(get("system.temp_critical_c", 80.0))
        self.cpu_warn = float(get("system.cpu_warn_pct", 80.0))
        self.cpu_critical = float(get("system.cpu_critical_pct", 92.0))
        self.ram_warn = float(get("system.ram_warn_pct", 80.0))
        self.ram_critical = float(get("system.ram_critical_pct", 92.0))

        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._prev_cpu: Optional[tuple[int, int]] = None  # (idle, total)
        self._snapshot: dict[str, Any] = {"ok": False, "ts": 0.0}

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if not self.enabled or (self._thread and self._thread.is_alive()):
            return
        # Prime the CPU delta so the first reported value is meaningful.
        self._read_cpu_percent()
        self.sample()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="voicepi-sysstats", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.sample()
            except Exception as exc:
                if self.logger is not None:
                    try:
                        self.logger.event("system", "error", f"sysstats sample failed: {exc}")
                    except Exception:
                        pass

    # -- readers -------------------------------------------------------------

    def _read_temp_c(self) -> Optional[float]:
        # Prefer the explicit CPU thermal zone, then fall back to the hottest zone.
        candidates = sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp")) if Path("/sys/class/thermal").exists() else []
        temps: list[float] = []
        for c in candidates:
            raw = _read_first_float(str(c))
            if raw is not None:
                # Values are in milli-degrees C on the Pi.
                temps.append(raw / 1000.0 if raw > 200 else raw)
        if temps:
            return round(max(temps), 1)
        return None

    def _read_mem(self) -> dict[str, Optional[float]]:
        info: dict[str, int] = {}
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip().split()[0]
                    info[key] = int(val)  # kB
        except Exception:
            return {"total_mb": None, "used_mb": None, "used_pct": None}
        total = info.get("MemTotal")
        avail = info.get("MemAvailable")
        if total is None or avail is None or total == 0:
            return {"total_mb": None, "used_mb": None, "used_pct": None}
        used = total - avail
        return {
            "total_mb": round(total / 1024.0, 1),
            "used_mb": round(used / 1024.0, 1),
            "used_pct": round(used / total * 100.0, 1),
        }

    def _read_cpu_percent(self) -> Optional[float]:
        try:
            first = Path("/proc/stat").read_text().splitlines()[0]
            fields = [int(x) for x in first.split()[1:]]
        except Exception:
            return None
        idle = fields[3] + (fields[4] if len(fields) > 4 else 0)  # idle + iowait
        total = sum(fields)
        prev = self._prev_cpu
        self._prev_cpu = (idle, total)
        if prev is None:
            return None
        idle_delta = idle - prev[0]
        total_delta = total - prev[1]
        if total_delta <= 0:
            return None
        return round((1.0 - idle_delta / total_delta) * 100.0, 1)

    def _read_loadavg(self) -> Optional[list[float]]:
        try:
            parts = Path("/proc/loadavg").read_text().split()
            return [float(parts[0]), float(parts[1]), float(parts[2])]
        except Exception:
            return None

    def _read_cpu_count(self) -> Optional[int]:
        try:
            import os
            return os.cpu_count()
        except Exception:
            return None

    def _read_cpu_freq_mhz(self) -> Optional[float]:
        raw = _read_first_float("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
        if raw is None:
            return None
        return round(raw / 1000.0, 0)  # kHz -> MHz

    def _read_uptime_s(self) -> Optional[float]:
        raw = _read_first_float("/proc/uptime")
        # /proc/uptime has two numbers; _read_first_float gets the whole line and fails.
        try:
            return round(float(Path("/proc/uptime").read_text().split()[0]), 0)
        except Exception:
            return raw

    # -- snapshot ------------------------------------------------------------

    def sample(self) -> dict[str, Any]:
        temp = self._read_temp_c()
        cpu = self._read_cpu_percent()
        mem = self._read_mem()
        load = self._read_loadavg()
        cores = self._read_cpu_count()
        ram_pct = mem.get("used_pct")

        # Express load average as a percentage of capacity (load / cores * 100).
        load_pct = None
        if load is not None and cores:
            load_pct = round(load[0] / cores * 100.0, 1)

        snapshot = {
            "ok": True,
            "ts": time.time(),
            "temperature": {
                "value_c": temp,
                "severity": _severity(temp, self.temp_warn, self.temp_critical),
                "warn": self.temp_warn,
                "critical": self.temp_critical,
            },
            "cpu": {
                "value_pct": cpu,
                "severity": _severity(cpu, self.cpu_warn, self.cpu_critical),
                "warn": self.cpu_warn,
                "critical": self.cpu_critical,
                "cores": cores,
                "freq_mhz": self._read_cpu_freq_mhz(),
                "load_avg": load,
                "load_pct": load_pct,
            },
            "ram": {
                "value_pct": ram_pct,
                "severity": _severity(ram_pct, self.ram_warn, self.ram_critical),
                "warn": self.ram_warn,
                "critical": self.ram_critical,
                "used_mb": mem.get("used_mb"),
                "total_mb": mem.get("total_mb"),
            },
            "uptime_s": self._read_uptime_s(),
        }
        # Overall severity is the worst of the three (an unknown metric never
        # downgrades a healthy system; it is treated as ok for the overall badge).
        order = {"ok": 0, "unknown": 0, "warn": 1, "critical": 2}
        worst = "ok"
        for s in (snapshot["temperature"]["severity"], snapshot["cpu"]["severity"], snapshot["ram"]["severity"]):
            if order.get(s, 0) > order.get(worst, 0):
                worst = s
        snapshot["severity"] = worst
        with self._lock:
            self._snapshot = snapshot
        return snapshot

    def latest(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._snapshot)