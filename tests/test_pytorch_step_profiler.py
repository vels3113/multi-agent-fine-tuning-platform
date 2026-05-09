"""Unit tests for bounded PyTorch step profiler aggregates."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.instrumentation.pytorch_step_profiler import (
    PytorchStepProfiler,
    parse_profiler_key_averages,
    should_profile_step,
    step_tracing_disabled_by_env,
)


def _ev(key: str, cuda_us: float = 0.0, cpu_us: float = 0.0):
    m = MagicMock()
    m.key = key
    m.self_cuda_time_total = cuda_us
    m.self_cpu_time_total = cpu_us
    return m


def test_should_profile_step_tenth_multiples():
    assert should_profile_step(1, 10) is True
    assert should_profile_step(10, 10) is True
    assert should_profile_step(11, 10) is False
    assert should_profile_step(2, 10) is False
    assert should_profile_step(1, 0) is False


def test_parse_profiler_keyword_split():
    out = parse_profiler_key_averages(
        [
            _ev("Custom.forward_block", cuda_us=3000),
            _ev("Custom.backward_block", cuda_us=5000),
            _ev("other", cuda_us=1000),
        ]
    )
    assert out["forward_ms"] == 3.0
    assert out["backward_ms"] == 5.0
    assert out["cuda_total_ms"] == 9.0
    assert out["profiler_note"] is None
    assert out["top_op_name"] == "Custom.backward_block"


def test_parse_profiler_fallback_cuda_total():
    out = parse_profiler_key_averages([_ev("aten::add", cuda_us=2000), _ev("aten::mm", cuda_us=5000)])
    assert out["forward_ms"] is None
    assert out["backward_ms"] is None
    assert out["cuda_total_ms"] == 7.0
    assert out["profiler_note"] == "forward_backward_unavailable"


def test_parse_profiler_cpu_only_note():
    out = parse_profiler_key_averages([_ev("cpu_op", cuda_us=0, cpu_us=1200)])
    assert out["cuda_total_ms"] is None
    assert out["profiler_note"] == "cpu_only"


def test_step_tracing_disabled_by_env(monkeypatch):
    monkeypatch.delenv("STEP_TRACING", raising=False)
    assert step_tracing_disabled_by_env() is False
    monkeypatch.setenv("STEP_TRACING", "0")
    assert step_tracing_disabled_by_env() is True
    monkeypatch.setenv("STEP_TRACING", "false")
    assert step_tracing_disabled_by_env() is True


def test_profiler_lifecycle_monkeypatch_torch():
    import torch

    mock_ctx = MagicMock()
    mock_ctx.key_averages.return_value = [_ev("my_forward", cuda_us=1500)]
    mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
    mock_ctx.__exit__ = MagicMock(return_value=None)

    with patch.object(torch.profiler, "profile", return_value=mock_ctx):
        ps = PytorchStepProfiler(profile_every_n=1)
        ps.on_rollout_start(1)
        agg = ps.on_loss_end(1)
        assert agg is not None
        assert agg["profiled"] is True
        assert agg["forward_ms"] == 1.5
        mock_ctx.__exit__.assert_called_once()


def test_profiler_no_cuda_activities_still_runs(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    import torch

    mock_ctx = MagicMock()
    mock_ctx.key_averages.return_value = []
    mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
    mock_ctx.__exit__ = MagicMock(return_value=None)

    with patch.object(torch.profiler, "profile", return_value=mock_ctx):
        ps = PytorchStepProfiler(profile_every_n=1)
        ps.on_rollout_start(1)
        agg = ps.on_loss_end(1)
        assert agg is not None
        assert agg["profiled"] is True
