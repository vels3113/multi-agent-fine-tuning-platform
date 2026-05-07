"""Tests for P1b single-agent baseline module."""
import pytest
from baseline.metrics import is_ast_parseable


# ── syntactic correctness helper ────────────────────────────────────────────

def test_is_ast_parseable_valid():
    assert is_ast_parseable("def f():\n    return 1") is True


def test_is_ast_parseable_invalid():
    assert is_ast_parseable("def f(\n    return 1") is False


# ── metrics schema ──────────────────────────────────────────────────────────

def test_metrics_schema():
    """Baseline JSON must contain exactly these top-level keys."""
    from baseline.metrics import METRICS_SCHEMA
    required = {"test_pass_rate", "pass@1", "pass@2", "pass@5",
                "syntactic_correctness_ratio", "token_throughput_per_sec",
                "num_problems", "num_runs", "model", "dataset", "timestamp"}
    assert required.issubset(set(METRICS_SCHEMA))


# ── token throughput ─────────────────────────────────────────────────────────

def test_token_throughput_positive():
    from baseline.metrics import compute_token_throughput
    tput = compute_token_throughput(total_tokens=1000, elapsed_seconds=5.0)
    assert tput == pytest.approx(200.0)


def test_token_throughput_zero_time_raises():
    from baseline.metrics import compute_token_throughput
    with pytest.raises(ZeroDivisionError):
        compute_token_throughput(total_tokens=100, elapsed_seconds=0.0)


# ── syntactic correctness ratio ──────────────────────────────────────────────

def test_syntactic_ratio_all_valid():
    from baseline.metrics import compute_syntactic_ratio
    samples = ["def f():\n    return 1", "x = 1 + 2"]
    assert compute_syntactic_ratio(samples) == pytest.approx(1.0)


def test_syntactic_ratio_mixed():
    from baseline.metrics import compute_syntactic_ratio
    samples = ["def f():\n    return 1", "def f(\n    return BAD"]
    assert compute_syntactic_ratio(samples) == pytest.approx(0.5)


def test_syntactic_ratio_empty_raises():
    from baseline.metrics import compute_syntactic_ratio
    with pytest.raises(ValueError):
        compute_syntactic_ratio([])
