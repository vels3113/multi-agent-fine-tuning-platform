import os
import pytest
from unittest.mock import MagicMock, patch


def test_init_disabled_by_env_var(monkeypatch):
    monkeypatch.setenv("WANDB_DISABLED", "1")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    from src.instrumentation.wandb_logger import WandbLogger
    logger = WandbLogger()
    run_id = logger.init(run_config={"seed": 42})
    assert run_id is None
    assert not logger._enabled


def test_init_disabled_when_no_api_key(monkeypatch):
    monkeypatch.delenv("WANDB_DISABLED", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    from src.instrumentation.wandb_logger import WandbLogger
    logger = WandbLogger(cfg={})
    run_id = logger.init(run_config={"seed": 42})
    assert run_id is None
    assert not logger._enabled


def test_init_disabled_when_wandb_not_installed(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "test-key-abc")
    monkeypatch.delenv("WANDB_DISABLED", raising=False)
    import sys
    original = sys.modules.get("wandb")
    sys.modules["wandb"] = None  # simulate ImportError
    try:
        from src.instrumentation.wandb_logger import WandbLogger
        logger = WandbLogger()
        run_id = logger.init(run_config={"seed": 42})
        assert run_id is None
    finally:
        if original is not None:
            sys.modules["wandb"] = original
        else:
            sys.modules.pop("wandb", None)


def test_log_noop_when_disabled(monkeypatch):
    monkeypatch.setenv("WANDB_DISABLED", "1")
    from src.instrumentation.wandb_logger import WandbLogger
    logger = WandbLogger()
    logger.init(run_config={})
    logger.log({"train/loss": 0.5}, step=1)  # must not raise


def test_finish_noop_when_disabled(monkeypatch):
    monkeypatch.setenv("WANDB_DISABLED", "1")
    from src.instrumentation.wandb_logger import WandbLogger
    logger = WandbLogger()
    logger.finish()  # must not raise


def test_run_id_none_when_disabled(monkeypatch):
    monkeypatch.setenv("WANDB_DISABLED", "1")
    from src.instrumentation.wandb_logger import WandbLogger
    logger = WandbLogger()
    logger.init(run_config={})
    assert logger.run_id is None


def test_init_enabled_calls_wandb_init(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    monkeypatch.delenv("WANDB_DISABLED", raising=False)
    monkeypatch.delenv("WANDB_RUN_ID", raising=False)

    mock_run = MagicMock()
    mock_run.id = "run-abc123"
    mock_wandb = MagicMock()
    mock_wandb.init.return_value = mock_run

    import sys
    sys.modules["wandb"] = mock_wandb
    try:
        import importlib
        import src.instrumentation.wandb_logger as wl
        importlib.reload(wl)
        logger = wl.WandbLogger(cfg={"project": "test-proj", "entity": "test-entity"})
        run_id = logger.init(run_config={"seed": 42})
        assert run_id == "run-abc123"
        assert logger._enabled
        mock_wandb.init.assert_called_once()
        call_kwargs = mock_wandb.init.call_args[1]
        assert call_kwargs["project"] == "test-proj"
    finally:
        sys.modules.pop("wandb", None)


def test_log_calls_wandb_log_when_enabled(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    monkeypatch.delenv("WANDB_DISABLED", raising=False)
    monkeypatch.delenv("WANDB_RUN_ID", raising=False)

    mock_run = MagicMock()
    mock_run.id = "run-xyz"
    mock_wandb = MagicMock()
    mock_wandb.init.return_value = mock_run

    import sys
    sys.modules["wandb"] = mock_wandb
    try:
        import importlib
        import src.instrumentation.wandb_logger as wl
        importlib.reload(wl)
        logger = wl.WandbLogger(cfg={"project": "test-proj"})
        logger.init(run_config={})
        logger.log({"train/loss": 0.42, "train/gpu_util_pct": 95.0}, step=5)
        mock_run.log.assert_called_once_with({"train/loss": 0.42, "train/gpu_util_pct": 95.0}, step=5)
    finally:
        sys.modules.pop("wandb", None)
