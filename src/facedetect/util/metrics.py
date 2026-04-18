from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass

import psutil

log = logging.getLogger(__name__)


@dataclass
class MetricsSnapshot:
    cpu_proc_pct: float       # this process, 0-100 (normalized by core count)
    cpu_system_pct: float     # whole-system CPU, 0-100
    mem_proc_mb: float        # resident set size of this process, MB
    mem_system_pct: float     # system memory used, 0-100
    mem_system_total_mb: float
    gpu_pct: float | None     # this process's GPU engine utilization, 0-100
    gpu_mem_mb: float | None  # this process's GPU dedicated memory, MB
    gpu_available: bool       # True if we were able to poll GPU counters


class SystemMetrics:
    """Background sampler for CPU / memory / GPU usage.

    CPU + memory are polled with psutil (cheap). GPU is polled via the Windows
    `typeperf` utility in a separate thread because a single invocation takes
    ~2 s — we cache the last reading between calls.
    """

    def __init__(self, *, gpu_poll_interval: float = 2.5):
        self._pid = os.getpid()
        self._proc = psutil.Process(self._pid)
        self._proc.cpu_percent(None)  # prime the sampler

        self._cpu_count = psutil.cpu_count(logical=True) or 1
        self._mem_total_mb = psutil.virtual_memory().total / (1024 * 1024)

        self._gpu_poll_interval = gpu_poll_interval
        self._gpu_lock = threading.Lock()
        self._gpu_pct: float | None = None
        self._gpu_mem_mb: float | None = None
        self._gpu_available = True  # optimistic; flips false after first failure
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._gpu_loop, name="gpu-metrics", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # --- public snapshot ---

    def snapshot(self) -> MetricsSnapshot:
        # Process CPU% — psutil returns percent of a single core, so divide by
        # core count to present "% of machine" which is what users expect.
        cpu_raw = self._proc.cpu_percent(None)
        cpu_proc = cpu_raw / self._cpu_count
        cpu_sys = psutil.cpu_percent(None)
        mem_info = self._proc.memory_info()
        vmem = psutil.virtual_memory()
        with self._gpu_lock:
            gpu_pct = self._gpu_pct
            gpu_mem_mb = self._gpu_mem_mb
            gpu_available = self._gpu_available
        return MetricsSnapshot(
            cpu_proc_pct=min(100.0, cpu_proc),
            cpu_system_pct=cpu_sys,
            mem_proc_mb=mem_info.rss / (1024 * 1024),
            mem_system_pct=vmem.percent,
            mem_system_total_mb=self._mem_total_mb,
            gpu_pct=gpu_pct,
            gpu_mem_mb=gpu_mem_mb,
            gpu_available=gpu_available,
        )

    # --- GPU sampler ---

    def _gpu_loop(self) -> None:
        while not self._stop.is_set():
            try:
                pct, mem_mb = self._sample_gpu_once()
            except FileNotFoundError:
                log.info("typeperf not available; GPU metrics disabled")
                with self._gpu_lock:
                    self._gpu_available = False
                return
            except Exception as e:  # noqa: BLE001
                log.debug("GPU sample failed: %s", e)
                pct, mem_mb = None, None
            with self._gpu_lock:
                self._gpu_pct = pct
                self._gpu_mem_mb = mem_mb
            if self._stop.wait(self._gpu_poll_interval):
                return

    _PID_RE = re.compile(r"pid_(\d+)_")

    def _sample_gpu_once(self) -> tuple[float | None, float | None]:
        """Run `typeperf` once and parse GPU util + dedicated memory for our PID."""
        cmd = [
            "typeperf",
            r"\GPU Engine(*)\Utilization Percentage",
            r"\GPU Process Memory(*)\Dedicated Usage",
            "-sc",
            "1",
        ]
        # CREATE_NO_WINDOW = 0x08000000 so the console window doesn't flash.
        creationflags = 0x08000000 if os.name == "nt" else 0
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=6,
            creationflags=creationflags,
        )
        if result.returncode != 0:
            return (None, None)
        return self._parse_typeperf(result.stdout, self._pid)

    @classmethod
    def _parse_typeperf(cls, stdout: str, pid: int) -> tuple[float | None, float | None]:
        # typeperf CSV layout: first line is the header with column names,
        # second line is the sample row.
        lines = [ln for ln in stdout.splitlines() if ln.strip()]
        if len(lines) < 2:
            return (None, None)
        header = _split_csv(lines[0])
        values = _split_csv(lines[1])
        if len(header) != len(values):
            return (None, None)

        total_util = 0.0
        total_mem = 0.0
        saw_util = False
        saw_mem = False
        pid_tag = f"pid_{pid}_"
        for name, raw in zip(header[1:], values[1:]):  # skip timestamp col
            if pid_tag not in name:
                continue
            try:
                v = float(raw)
            except ValueError:
                continue
            if "GPU Engine" in name:
                total_util += v
                saw_util = True
            elif "GPU Process Memory" in name:
                total_mem += v
                saw_mem = True
        util = min(100.0, total_util) if saw_util else None
        mem_mb = (total_mem / (1024 * 1024)) if saw_mem else None
        return (util, mem_mb)


def _split_csv(line: str) -> list[str]:
    """Minimal CSV splitter for typeperf output (double-quoted fields)."""
    out: list[str] = []
    buf: list[str] = []
    in_q = False
    for ch in line:
        if ch == '"':
            in_q = not in_q
            continue
        if ch == "," and not in_q:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    out.append("".join(buf))
    return out
