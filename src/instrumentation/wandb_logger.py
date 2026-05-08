"""W&B per-step training logger. Gracefully disabled if wandb unavailable or not configured."""
import logging
import os

logger = logging.getLogger(__name__)


class WandbLogger:
    """Optional W&B run wrapper.

    Disabled (all methods are no-ops) when:
    - WANDB_DISABLED=1 environment variable is set
    - WANDB_API_KEY is absent and not in cfg
    - wandb package is not installed

    When enabled, logs per-step training and hardware metrics to W&B.
    """

    def __init__(self, cfg: dict | None = None):
        self._run = None
        self._enabled = False
        self._cfg = cfg or {}

    def init(self, run_config: dict) -> str | None:
        """Initialize W&B run. Returns run ID string or None when disabled."""
        if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes"):
            logger.info("W&B disabled (WANDB_DISABLED=%s)", os.environ["WANDB_DISABLED"])
            return None

        try:
            import wandb
            if wandb is None:
                raise ImportError("wandb is None")
        except (ImportError, TypeError):
            logger.warning("wandb not installed — W&B logging disabled")
            return None

        api_key = os.environ.get("WANDB_API_KEY") or self._cfg.get("api_key")
        if not api_key:
            logger.warning("WANDB_API_KEY not set — W&B logging disabled")
            return None

        try:
            self._run = wandb.init(
                project=self._cfg.get("project") or os.environ.get("WANDB_PROJECT"),
                entity=self._cfg.get("entity") or os.environ.get("WANDB_ENTITY"),
                name=self._cfg.get("run_name"),
                config=run_config,
                resume="allow",
                id=os.environ.get("WANDB_RUN_ID") or None,
            )
            self._enabled = True
            logger.info("W&B run initialized: %s", self._run.id)
            return self._run.id
        except Exception as exc:
            logger.warning("W&B init failed: %s — continuing without W&B", exc)
            return None

    def log(self, metrics: dict, step: int) -> None:
        """Log a dict of metrics at the given step. No-op when disabled."""
        if not self._enabled or self._run is None:
            return
        try:
            self._run.log(metrics, step=step)
        except Exception as exc:
            logger.warning("W&B log failed at step %d: %s", step, exc)

    def finish(self) -> None:
        """Finish the W&B run. No-op when disabled."""
        if not self._enabled or self._run is None:
            return
        try:
            self._run.finish()
        except Exception as exc:
            logger.warning("W&B finish failed: %s", exc)
        finally:
            self._enabled = False
            self._run = None

    @property
    def run_id(self) -> str | None:
        return self._run.id if self._run is not None else None
