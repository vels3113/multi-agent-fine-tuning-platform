import time
import pytest
from unittest.mock import patch, MagicMock


# ── Parser unit tests (no subprocess) ────────────────────────────────────────

def test_parse_gpu_util_standard_key():
    from src.instrumentation.smi_poller import _parse_gpu_util
    data = {"card0": {"GPU use (%)": "95"}}
    assert _parse_gpu_util(data) == 95.0


def test_parse_gpu_util_alt_key():
    from src.instrumentation.smi_poller import _parse_gpu_util
    data = {"card0": {"GPU Use (%)": "42"}}
    assert _parse_gpu_util(data) == 42.0


def test_parse_gpu_util_missing_returns_none():
    from src.instrumentation.smi_poller import _parse_gpu_util
    assert _parse_gpu_util({"card0": {}}) is None
    assert _parse_gpu_util({}) is None


def test_parse_vram_used_mb_from_bytes():
    from src.instrumentation.smi_poller import _parse_vram_used_mb
    data = {"card0": {"VRAM Total Used Memory (B)": str(1024 ** 2 * 100)}}
    assert _parse_vram_used_mb(data) == pytest.approx(100.0)


def test_parse_vram_used_mb_missing_returns_none():
    from src.instrumentation.smi_poller import _parse_vram_used_mb
    assert _parse_vram_used_mb({"card0": {}}) is None


# ── _query_rocm_smi with subprocess mock ─────────────────────────────────────

def test_query_returns_none_when_rocm_smi_missing():
    from src.instrumentation.smi_poller import _query_rocm_smi
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert _query_rocm_smi() is None


def test_query_returns_none_on_nonzero_exit():
    from src.instrumentation.smi_poller import _query_rocm_smi
    result = MagicMock()
    result.returncode = 1
    result.stdout = ""
    with patch("subprocess.run", return_value=result):
        assert _query_rocm_smi() is None


def test_query_returns_parsed_json():
    from src.instrumentation.smi_poller import _query_rocm_smi
    result = MagicMock()
    result.returncode = 0
    result.stdout = '{"card0": {"GPU use (%)": "80"}}'
    with patch("subprocess.run", return_value=result):
        data = _query_rocm_smi()
    assert data == {"card0": {"GPU use (%)": "80"}}


# ── SmiPoller lifecycle ───────────────────────────────────────────────────────

def test_get_stats_returns_none_when_no_samples():
    from src.instrumentation.smi_poller import SmiPoller
    poller = SmiPoller(interval=60.0)
    stats = poller.get_stats()
    assert stats["gpu_utilization_pct_mean"] is None
    assert stats["vram_used_mb_mean"] is None
    assert stats["sample_count"] == 0


def test_get_stats_computes_mean():
    from src.instrumentation.smi_poller import SmiPoller
    poller = SmiPoller(interval=60.0)
    poller._gpu_util_history.extend([80.0, 90.0, 100.0])
    poller._vram_used_history.extend([50000.0, 60000.0, 55000.0])
    stats = poller.get_stats()
    assert stats["gpu_utilization_pct_mean"] == pytest.approx(90.0, rel=0.01)
    assert stats["vram_used_mb_mean"] == pytest.approx(55000.0, rel=0.01)
    assert stats["sample_count"] == 3


def test_poller_start_stop_no_crash_when_rocm_smi_unavailable():
    from src.instrumentation.smi_poller import SmiPoller
    with patch("src.instrumentation.smi_poller._query_rocm_smi", return_value=None):
        poller = SmiPoller(interval=0.05)
        poller.start()
        time.sleep(0.15)
        poller.stop()
        stats = poller.get_stats()
    assert stats["gpu_utilization_pct_mean"] is None
    assert stats["sample_count"] == 0


def test_poller_accumulates_samples():
    from src.instrumentation.smi_poller import SmiPoller
    mock_data = {"card0": {"GPU use (%)": "75", "VRAM Total Used Memory (B)": str(1024 ** 2 * 30000)}}
    with patch("src.instrumentation.smi_poller._query_rocm_smi", return_value=mock_data):
        poller = SmiPoller(interval=0.05)
        poller.start()
        time.sleep(0.25)
        poller.stop()
        stats = poller.get_stats()
    assert stats["sample_count"] >= 2
    assert stats["gpu_utilization_pct_mean"] == pytest.approx(75.0, rel=0.01)
