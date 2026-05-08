"""Per-step training guards. Each raises ValueError if the step is unsafe."""


def check_loss(loss: float, step: int) -> None:
    import math
    if math.isnan(loss) or math.isinf(loss):
        raise ValueError(f"NaN/Inf loss detected at step {step}: {loss}")


def check_kl(current_kl: float, baseline_kl: float | None,
             threshold_multiplier: float, step: int) -> None:
    if baseline_kl is None:
        return
    if current_kl > baseline_kl * threshold_multiplier:
        raise ValueError(
            f"KL spike at step {step}: current={current_kl:.4f} "
            f"baseline={baseline_kl:.4f} threshold_multiplier={threshold_multiplier}"
        )


def check_reward_collapse(rewards: list[float], step: int,
                           min_std: float = 1e-4) -> None:
    if not rewards:
        return
    mean = sum(rewards) / len(rewards)
    variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = variance ** 0.5
    if std < min_std:
        raise ValueError(
            f"reward collapse at step {step}: std={std:.6f} rewards={rewards[:4]}..."
        )
