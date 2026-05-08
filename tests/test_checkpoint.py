import os
import tempfile
import torch
import pytest
from unittest.mock import MagicMock, patch


def _make_trainer_mock():
    trainer = MagicMock()
    trainer.model.state_dict.return_value = {"w": torch.tensor(1.0)}
    trainer.optimizer.state_dict.return_value = {"step": 1}
    trainer.lr_scheduler = None
    trainer.get_dataloader_state = MagicMock(return_value={"idx": 0})
    trainer.rollout_buffer = MagicMock()
    trainer.rollout_buffer.__len__ = lambda self: 0
    trainer.rollout_buffer.__iter__ = lambda self: iter([])
    return trainer


def test_save_creates_checkpoint_dir():
    from checkpoint import CheckpointManager
    with tempfile.TemporaryDirectory() as tmpdir:
        cm = CheckpointManager(tmpdir, keep=2)
        trainer = _make_trainer_mock()
        cm.save(trainer, step=1, wandb_run_id="run-abc")
        assert os.path.isdir(os.path.join(tmpdir, "ckpt-000001"))


def test_save_writes_metadata_json():
    from checkpoint import CheckpointManager
    import json
    with tempfile.TemporaryDirectory() as tmpdir:
        cm = CheckpointManager(tmpdir, keep=2)
        trainer = _make_trainer_mock()
        cm.save(trainer, step=1, wandb_run_id="run-abc")
        meta_path = os.path.join(tmpdir, "ckpt-000001", "meta.json")
        assert os.path.exists(meta_path)
        meta = json.loads(open(meta_path).read())
        assert meta["step"] == 1
        assert meta["wandb_run_id"] == "run-abc"


def test_rotate_keeps_last_two():
    from checkpoint import CheckpointManager
    with tempfile.TemporaryDirectory() as tmpdir:
        cm = CheckpointManager(tmpdir, keep=2)
        trainer = _make_trainer_mock()
        cm.save(trainer, step=1, wandb_run_id=None)
        cm.save(trainer, step=2, wandb_run_id=None)
        cm.save(trainer, step=3, wandb_run_id=None)
        dirs = sorted(os.listdir(tmpdir))
        assert dirs == ["ckpt-000002", "ckpt-000003"]


def test_latest_checkpoint_path():
    from checkpoint import CheckpointManager
    with tempfile.TemporaryDirectory() as tmpdir:
        cm = CheckpointManager(tmpdir, keep=2)
        assert cm.latest_path() is None
        trainer = _make_trainer_mock()
        cm.save(trainer, step=5, wandb_run_id=None)
        assert cm.latest_path() == os.path.join(tmpdir, "ckpt-000005")


def test_load_restores_step_and_wandb_id():
    from checkpoint import CheckpointManager
    with tempfile.TemporaryDirectory() as tmpdir:
        cm = CheckpointManager(tmpdir, keep=2)
        trainer = _make_trainer_mock()
        cm.save(trainer, step=7, wandb_run_id="run-xyz")
        state = cm.load(cm.latest_path(), trainer)
        assert state["step"] == 7
        assert state["wandb_run_id"] == "run-xyz"
