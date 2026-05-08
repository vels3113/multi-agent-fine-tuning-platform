"""Full-state checkpoint save/load/rotate. Keep last N checkpoints."""
import json
import os
import re
import shutil
import torch


class CheckpointManager:
    def __init__(self, checkpoint_dir: str, keep: int = 2):
        self.dir = checkpoint_dir
        self.keep = keep
        os.makedirs(self.dir, exist_ok=True)

    def save(self, trainer, step: int, wandb_run_id: str | None) -> str:
        name = f"ckpt-{step:06d}"
        ckpt_path = os.path.join(self.dir, name)
        tmp_path = ckpt_path + ".tmp"
        os.makedirs(tmp_path, exist_ok=True)

        # Support both MAGRPOTrainer (.agents list) and mock trainers (.model/.optimizer)
        if hasattr(trainer, "agents"):
            agents_state = [a.state_dict() for a in trainer.agents]
            torch.save(agents_state, os.path.join(tmp_path, "model.pt"))
        else:
            torch.save(trainer.model.state_dict(), os.path.join(tmp_path, "model.pt"))

        if hasattr(trainer, "optimizers"):
            optims_state = [o.state_dict() for o in trainer.optimizers]
            torch.save(optims_state, os.path.join(tmp_path, "optimizer.pt"))
        else:
            torch.save(trainer.optimizer.state_dict(), os.path.join(tmp_path, "optimizer.pt"))

        if not hasattr(trainer, "agents") and getattr(trainer, "lr_scheduler", None) is not None:
            torch.save(trainer.lr_scheduler.state_dict(), os.path.join(tmp_path, "scheduler.pt"))

        rng_state = {
            "python": None,
            "numpy": None,
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }
        torch.save(rng_state, os.path.join(tmp_path, "rng.pt"))

        dl_state = trainer.get_dataloader_state() if callable(getattr(trainer, "get_dataloader_state", None)) else {}
        torch.save(dl_state, os.path.join(tmp_path, "dataloader.pt"))

        if hasattr(trainer, "rollout_buffers"):
            rb = [list(buf) for buf in trainer.rollout_buffers] if trainer.rollout_buffers else []
        elif hasattr(trainer, "rollout_buffer") and trainer.rollout_buffer is not None:
            rb = list(trainer.rollout_buffer)
        else:
            rb = []
        torch.save(rb, os.path.join(tmp_path, "rollout_buffer.pt"))

        meta = {"step": step, "wandb_run_id": wandb_run_id}
        with open(os.path.join(tmp_path, "meta.json"), "w") as f:
            json.dump(meta, f)

        os.rename(tmp_path, ckpt_path)
        self._rotate()
        return ckpt_path

    def load(self, ckpt_path: str, trainer) -> dict:
        meta_path = os.path.join(ckpt_path, "meta.json")
        with open(meta_path) as f:
            meta = json.load(f)

        model_data = torch.load(os.path.join(ckpt_path, "model.pt"), weights_only=True)
        if hasattr(trainer, "agents"):
            for agent, state in zip(trainer.agents, model_data):
                agent.load_state_dict(state)
        else:
            trainer.model.load_state_dict(model_data)

        optim_data = torch.load(os.path.join(ckpt_path, "optimizer.pt"), weights_only=True)
        if hasattr(trainer, "optimizers"):
            for optim, state in zip(trainer.optimizers, optim_data):
                optim.load_state_dict(state)
        else:
            trainer.optimizer.load_state_dict(optim_data)

        sched_path = os.path.join(ckpt_path, "scheduler.pt")
        if os.path.exists(sched_path) and getattr(trainer, "lr_scheduler", None) is not None:
            trainer.lr_scheduler.load_state_dict(torch.load(sched_path, weights_only=True))

        rng = torch.load(os.path.join(ckpt_path, "rng.pt"), weights_only=False)
        torch.set_rng_state(rng["torch"])
        if rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng["cuda"])

        return meta

    def latest_path(self) -> str | None:
        entries = self._sorted_checkpoints()
        return entries[-1] if entries else None

    def _sorted_checkpoints(self) -> list[str]:
        pattern = re.compile(r"^ckpt-\d{6}$")
        entries = [
            os.path.join(self.dir, d)
            for d in os.listdir(self.dir)
            if pattern.match(d) and os.path.isdir(os.path.join(self.dir, d))
        ]
        return sorted(entries)

    def _rotate(self):
        entries = self._sorted_checkpoints()
        for old in entries[: -self.keep]:
            shutil.rmtree(old, ignore_errors=True)
