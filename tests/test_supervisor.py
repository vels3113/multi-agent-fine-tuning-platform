import time
import pytest
from unittest.mock import MagicMock, patch


def test_rocm_hang_detected():
    from supervisor import StallDetector
    det = StallDetector(expected_workers=2, poll_interval=0.1, hang_threshold=0.5)
    snap = {"watchdog_ts": time.time() - 1.0,
            "cuda_allocated_mb": 100.0, "cuda_reserved_mb": 110.0,
            "workers": 2, "batch": 5}
    assert det.check(snap) == "rocm_hang"


def test_no_stall_when_healthy():
    from supervisor import StallDetector
    det = StallDetector(expected_workers=2, poll_interval=0.1, hang_threshold=5.0)
    snap = {"watchdog_ts": time.time(),
            "cuda_allocated_mb": 100.0, "cuda_reserved_mb": 110.0,
            "workers": 2, "batch": 5}
    assert det.check(snap) is None


def test_worker_death_detected():
    from supervisor import StallDetector
    det = StallDetector(expected_workers=2, poll_interval=0.1, hang_threshold=5.0)
    snap = {"watchdog_ts": time.time(),
            "cuda_allocated_mb": 100.0, "cuda_reserved_mb": 110.0,
            "workers": 1, "batch": 5}
    assert det.check(snap) == "worker_death"


def test_hbm_stall_detected_after_3_flat_polls():
    from supervisor import StallDetector
    det = StallDetector(expected_workers=2, poll_interval=0.1, hang_threshold=5.0)
    snap = {"watchdog_ts": time.time(),
            "cuda_allocated_mb": 100.0, "cuda_reserved_mb": 110.0,
            "workers": 2, "batch": 5}
    assert det.check(snap) is None
    assert det.check(snap) is None
    assert det.check(snap) == "hbm_stall"


def test_hbm_stall_resets_on_activity():
    from supervisor import StallDetector
    det = StallDetector(expected_workers=2, poll_interval=0.1, hang_threshold=5.0)
    snap_flat = {"watchdog_ts": time.time(),
                 "cuda_allocated_mb": 100.0, "cuda_reserved_mb": 110.0,
                 "workers": 2, "batch": 5}
    det.check(snap_flat)
    det.check(snap_flat)
    snap_active = {**snap_flat, "cuda_allocated_mb": 200.0}
    assert det.check(snap_active) is None
    assert det.check(snap_flat) is None


def test_retry_limit():
    from supervisor import SupervisorState
    state = SupervisorState(max_retries=3)
    assert state.should_retry()
    state.record_attempt()
    state.record_attempt()
    state.record_attempt()
    assert not state.should_retry()
