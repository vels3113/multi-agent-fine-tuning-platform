"""Run session storage — records config, metrics, and runtime attributes per run."""
import datetime
import json
import logging
import os
import socket
import time
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Session:
    session_id: str
    timestamp: str
    user: str | None
    stage: dict           # {"baseline": bool, "training": bool}
    config: dict
    metrics: dict
    runtime: dict         # partially filled at start, completed at update()
    _t0: float = field(default=0.0, init=False, repr=False)

    @classmethod
    def start(cls, config: dict, stage: dict, user: str | None = None) -> "Session":
        try:
            import torch
        except ImportError:
            num_gpus = 0
        else:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            num_gpus = torch.cuda.device_count()

        instance = cls(
            session_id=str(uuid.uuid4()),
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
            user=user,
            stage=stage,
            config=config,
            metrics={},
            runtime={
                "hostname": socket.gethostname(),
                "num_gpus": num_gpus,
                "peak_gpu_memory_mb": None,
                "gpu_utilization_pct": None,  # deferred to P3a (rocm-smi integration)
                "total_duration_sec": None,
            },
        )
        instance._t0 = time.perf_counter()
        return instance

    @classmethod
    def load(cls, session_id: str, sessions_dir: str) -> "Session":
        """Resume a prior session (e.g. after a checkpoint recovery).

        Reads the existing JSON, resets the peak memory stats, and restarts
        the wall-clock timer so total_duration_sec accumulates correctly on
        the next update() call.
        """
        path = os.path.join(sessions_dir, f"{session_id}.json")
        with open(path) as f:
            data = json.load(f)

        try:
            import torch
        except ImportError:
            pass
        else:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        instance = cls(
            session_id=data["session_id"],
            timestamp=data["timestamp"],
            user=data.get("user"),
            stage=data["stage"],
            config=data["config"],
            metrics=data.get("metrics", {}),
            runtime=data["runtime"],
        )
        instance._t0 = time.perf_counter()
        return instance

    def update(self, metrics: dict, sessions_dir: str) -> None:
        elapsed = time.perf_counter() - self._t0
        self.metrics = metrics

        try:
            import torch
        except ImportError:
            pass
        else:
            if torch.cuda.is_available():
                peak_bytes = torch.cuda.max_memory_allocated()
                self.runtime["peak_gpu_memory_mb"] = round(peak_bytes / 1024 ** 2, 2)

        self.runtime["total_duration_sec"] = round(elapsed, 3)

        os.makedirs(sessions_dir, exist_ok=True)
        out_path = os.path.join(sessions_dir, f"{self.session_id}.json")
        tmp_path = out_path + ".tmp"
        data = {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "user": self.user,
            "stage": self.stage,
            "config": self.config,
            "metrics": self.metrics,
            "runtime": self.runtime,
        }
        try:
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, out_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        logger.info("Session written to %s", out_path)
