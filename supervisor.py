"""Supervisor: launches trainer subprocess, detects crash/stall, auto-resumes with capped retries."""
import logging
import os
import signal
import subprocess
import sys
import time

from watchdog import read_snapshot

logger = logging.getLogger(__name__)


class SupervisorState:
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self._attempts = 0

    def should_retry(self) -> bool:
        return self._attempts < self.max_retries

    def record_attempt(self) -> None:
        self._attempts += 1

    @property
    def attempts(self) -> int:
        return self._attempts


class StallDetector:
    def __init__(self, expected_workers: int, poll_interval: float,
                 hang_threshold: float, hbm_flat_limit: int = 3,
                 hbm_delta_threshold: float = 1.0):
        self.expected_workers = expected_workers
        self.hang_threshold = hang_threshold
        self.hbm_flat_limit = hbm_flat_limit
        self.hbm_delta_threshold = hbm_delta_threshold
        self._prev_alloc: float | None = None
        self._flat_count = 0

    def check(self, snap: dict) -> str | None:
        now = time.time()

        if now - snap["watchdog_ts"] > self.hang_threshold:
            return "rocm_hang"

        if snap["workers"] < self.expected_workers:
            return "worker_death"

        if snap["cuda_reserved_mb"] > 0:
            if self._prev_alloc is None or abs(snap["cuda_allocated_mb"] - self._prev_alloc) < self.hbm_delta_threshold:
                self._flat_count += 1
            else:
                self._flat_count = 0
            self._prev_alloc = snap["cuda_allocated_mb"]

        if self._flat_count >= self.hbm_flat_limit:
            return "hbm_stall"

        return None


def run(trainer_cmd: list[str], shm_name: str, expected_workers: int,
        max_retries: int = 3, poll_interval: float = 5.0,
        hang_threshold: float = 10.0, heartbeat_path: str | None = None) -> int:
    """
    Launch trainer_cmd as subprocess. Poll shared memory for stalls.
    On crash or stall: kill, increment retry counter, re-launch with --resume-session.
    Returns exit code of the final run (0 = success).
    """
    state = SupervisorState(max_retries=max_retries)
    detector = StallDetector(expected_workers=expected_workers,
                             poll_interval=poll_interval,
                             hang_threshold=hang_threshold)
    cmd = list(trainer_cmd)

    while True:
        state.record_attempt()
        logger.info("Supervisor: starting trainer (attempt %d/%d): %s",
                    state.attempts, max_retries, cmd)
        proc = subprocess.Popen(cmd)

        exit_code = None
        while exit_code is None:
            time.sleep(poll_interval)

            if heartbeat_path:
                _write_heartbeat(heartbeat_path, state.attempts, max_retries)

            exit_code = proc.poll()
            if exit_code is not None:
                break

            try:
                snap = read_snapshot(shm_name)
            except FileNotFoundError:
                # shared memory not yet created by trainer — still starting up
                continue

            stall_reason = detector.check(snap)
            if stall_reason:
                logger.warning("Supervisor: stall detected (%s) — killing trainer", stall_reason)
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                exit_code = -1
                break

        if exit_code == 0:
            logger.info("Supervisor: trainer exited cleanly.")
            return 0

        if not state.should_retry():
            logger.error("Supervisor: max retries (%d) reached. Giving up.", max_retries)
            return exit_code

        logger.warning("Supervisor: trainer failed (code=%d). Retrying...", exit_code)
        # Signal via environment variable so trainer picks up the latest checkpoint.
        os.environ["SUPERVISOR_RESUME"] = "1"


def _write_heartbeat(path: str, attempt: int, max_retries: int) -> None:
    import json
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"ts": time.time(), "attempt": attempt, "max_retries": max_retries}, f)
    os.replace(tmp, path)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--shm-name", required=True)
    parser.add_argument("--expected-workers", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--hang-threshold", type=float, default=10.0)
    parser.add_argument("--heartbeat-path", default=None)
    parser.add_argument("trainer_cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    sys.exit(run(
        trainer_cmd=args.trainer_cmd,
        shm_name=args.shm_name,
        expected_workers=args.expected_workers,
        max_retries=args.max_retries,
        poll_interval=args.poll_interval,
        hang_threshold=args.hang_threshold,
        heartbeat_path=args.heartbeat_path,
    ))
