"""Tests for platform/session.py"""
import json
import os
import uuid
import tempfile
import pytest


def test_session_start_has_required_keys():
    from session import Session
    s = Session.start(
        config={"model": "m", "dataset": "d", "num_problems": 3,
                "num_runs": 1, "batch_size": 1, "max_new_tokens": 512,
                "stop_sequences": [], "seed": 42},
        stage={"baseline": True, "training": False},
    )
    assert isinstance(s.session_id, str)
    uuid.UUID(s.session_id)  # raises if not valid uuid4
    assert s.timestamp.endswith("Z")
    assert s.user is None
    assert s.stage == {"baseline": True, "training": False}
    assert "model" in s.config
    assert s.metrics == {}
    assert "hostname" in s.runtime
    assert "num_gpus" in s.runtime


def test_session_update_writes_json():
    from session import Session
    s = Session.start(
        config={"model": "m", "dataset": "d", "num_problems": 3,
                "num_runs": 1, "batch_size": 1, "max_new_tokens": 512,
                "stop_sequences": [], "seed": 42},
        stage={"baseline": True, "training": False},
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        s.update(
            metrics={"test_pass_rate": 0.1, "syntactic_correctness_ratio": 0.9,
                     "token_throughput_per_sec": 500.0},
            sessions_dir=tmpdir,
        )
        files = os.listdir(tmpdir)
        assert len(files) == 1
        assert files[0] == f"{s.session_id}.json"
        with open(os.path.join(tmpdir, files[0])) as f:
            data = json.load(f)
        assert data["session_id"] == s.session_id
        assert data["metrics"]["test_pass_rate"] == pytest.approx(0.1)
        assert data["runtime"]["total_duration_sec"] >= 0
        assert data["runtime"]["gpu_utilization_pct"] is None


def test_session_update_creates_sessions_dir():
    from session import Session
    s = Session.start(
        config={"model": "m", "dataset": "d", "num_problems": 1,
                "num_runs": 1, "batch_size": 1, "max_new_tokens": 128,
                "stop_sequences": [], "seed": 0},
        stage={"baseline": False, "training": True},
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        new_dir = os.path.join(tmpdir, "sessions", "nested")
        s.update(metrics={}, sessions_dir=new_dir)
        assert os.path.isdir(new_dir)
