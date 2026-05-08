"""Daemon watchdog thread — writes liveness + GPU state to shared memory every N seconds."""
import multiprocessing
import struct
import threading
import time
from multiprocessing.shared_memory import SharedMemory

STRUCT_FMT = ">dddii8x"
STRUCT_SIZE = struct.calcsize(STRUCT_FMT)  # 40 bytes


class Watchdog:
    def __init__(self, shm_name: str, interval: float = 2.0):
        self.shm_name = shm_name
        self.interval = interval
        self._shm: SharedMemory | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._batch_index = 0
        self._lock = threading.Lock()

    def start(self, initial_workers: int, initial_batch: int) -> None:
        self._shm = SharedMemory(name=self.shm_name, create=True, size=STRUCT_SIZE)
        self._batch_index = initial_batch
        self._thread = threading.Thread(
            target=self._run, args=(initial_workers,), daemon=True, name="watchdog"
        )
        self._thread.start()

    def update_batch(self, batch_index: int) -> None:
        with self._lock:
            self._batch_index = batch_index

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._shm is not None:
            self._shm.close()
            self._shm.unlink()
            self._shm = None

    def _run(self, initial_workers: int) -> None:
        while not self._stop_event.is_set():
            try:
                import torch
                alloc = torch.cuda.memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0
                reserved = torch.cuda.memory_reserved() / 1024 ** 2 if torch.cuda.is_available() else 0.0
            except Exception:
                alloc, reserved = 0.0, 0.0

            live = len(multiprocessing.active_children())
            workers = live if live > 0 else initial_workers
            with self._lock:
                batch = self._batch_index

            data = struct.pack(STRUCT_FMT, time.time(), alloc, reserved, workers, batch)
            self._shm.buf[:STRUCT_SIZE] = data
            self._stop_event.wait(self.interval)


def read_snapshot(shm_name: str) -> dict:
    shm = SharedMemory(name=shm_name, create=False)
    data = bytes(shm.buf[:STRUCT_SIZE])
    shm.close()
    ts, alloc, reserved, workers, batch = struct.unpack(STRUCT_FMT, data)
    return {
        "watchdog_ts": ts,
        "cuda_allocated_mb": alloc,
        "cuda_reserved_mb": reserved,
        "workers": workers,
        "batch": batch,
    }
