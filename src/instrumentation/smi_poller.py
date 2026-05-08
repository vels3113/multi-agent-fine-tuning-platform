"""rocm-smi background polling thread — captures GPU utilization and VRAM usage per training run."""
import json
import logging
import subprocess
import threading
import time
from collections import deque

logger = logging.getLogger(__name__)


def _query_rocm_smi() -> dict | None:
    """Run rocm-smi and return parsed JSON. Returns None if unavailable or error."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError, ValueError) as exc:
        logger.debug("rocm-smi query failed: %s", exc)
        return None


def _parse_gpu_util(data: dict) -> float | None:
    """Extract GPU utilization % from rocm-smi JSON (card0). Handles multiple field name conventions."""
    card = data.get("card0", {})
    for key in ("GPU use (%)", "GPU Use (%)", "gpu_use_pct", "GPU Utilization (%)"):
        if key in card:
            try:
                return float(str(card[key]).strip().rstrip("%"))
            except (ValueError, TypeError):
                pass
    return None


def _parse_vram_used_mb(data: dict) -> float | None:
    """Extract VRAM used MB from rocm-smi JSON (card0)."""
    card = data.get("card0", {})
    for key in ("VRAM Total Used Memory (B)", "vram_used_bytes"):
        if key in card:
            try:
                return int(card[key]) / 1024 ** 2
            except (ValueError, TypeError):
                pass
    return None


class SmiPoller:
    """Daemon thread that polls rocm-smi every `interval` seconds.

    Accumulates GPU utilization % and VRAM used MB samples for the duration
    of a training run.  Gracefully no-ops if rocm-smi is unavailable.
    """

    def __init__(self, interval: float = 5.0, history_size: int = 7200):
        self.interval = interval
        self._gpu_util_history: deque[float] = deque(maxlen=history_size)
        self._vram_used_history: deque[float] = deque(maxlen=history_size)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="smi-poller")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def get_stats(self) -> dict:
        """Return mean GPU util % and mean VRAM used MB over all collected samples.

        Returns None values when no samples were collected (rocm-smi unavailable).
        """
        util_samples = list(self._gpu_util_history)
        vram_samples = list(self._vram_used_history)
        return {
            "gpu_utilization_pct_mean": (
                round(sum(util_samples) / len(util_samples), 1) if util_samples else None
            ),
            "vram_used_mb_mean": (
                round(sum(vram_samples) / len(vram_samples), 1) if vram_samples else None
            ),
            "sample_count": len(util_samples),
        }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            data = _query_rocm_smi()
            if data is not None:
                util = _parse_gpu_util(data)
                vram = _parse_vram_used_mb(data)
                if util is not None:
                    self._gpu_util_history.append(util)
                if vram is not None:
                    self._vram_used_history.append(vram)
            self._stop_event.wait(self.interval)
