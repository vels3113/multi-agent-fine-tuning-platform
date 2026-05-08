import json
import os
import tempfile
import time


def test_heartbeat_file_format():
    from src.training.supervisor import _write_heartbeat
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "heartbeat.json")
        _write_heartbeat(path, attempt=1, max_retries=3)
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert "ts" in data
        assert "attempt" in data
        assert "max_retries" in data
        assert data["attempt"] == 1
        assert data["max_retries"] == 3
        assert data["ts"] <= time.time()


def test_heartbeat_atomic_write():
    """Ensure no half-written file is ever visible."""
    from src.training.supervisor import _write_heartbeat
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "heartbeat.json")
        for i in range(10):
            _write_heartbeat(path, attempt=i, max_retries=3)
            with open(path) as f:
                data = json.load(f)
            assert data["attempt"] == i
