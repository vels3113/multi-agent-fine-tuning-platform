import pytest


def test_nan_guard_raises_on_nan_loss():
    from guards import check_loss
    with pytest.raises(ValueError, match="NaN/Inf"):
        check_loss(float("nan"), step=1)


def test_nan_guard_raises_on_inf_loss():
    from guards import check_loss
    with pytest.raises(ValueError, match="NaN/Inf"):
        check_loss(float("inf"), step=1)


def test_nan_guard_passes_on_normal_loss():
    from guards import check_loss
    check_loss(0.42, step=1)


def test_kl_spike_guard_raises_when_exceeded():
    from guards import check_kl
    with pytest.raises(ValueError, match="KL spike"):
        check_kl(current_kl=5.0, baseline_kl=0.1, threshold_multiplier=10.0, step=1)


def test_kl_spike_guard_passes_when_within_threshold():
    from guards import check_kl
    check_kl(current_kl=0.5, baseline_kl=0.1, threshold_multiplier=10.0, step=1)


def test_kl_spike_guard_skips_when_no_baseline():
    from guards import check_kl
    check_kl(current_kl=99.0, baseline_kl=None, threshold_multiplier=10.0, step=1)


def test_reward_collapse_guard_raises():
    from guards import check_reward_collapse
    rewards = [0.0] * 8
    with pytest.raises(ValueError, match="reward collapse"):
        check_reward_collapse(rewards, step=1)


def test_reward_collapse_guard_passes_with_variance():
    from guards import check_reward_collapse
    rewards = [0.0, 1.0, 0.5, 0.8, 0.2, 0.9, 0.1, 0.7]
    check_reward_collapse(rewards, step=1)
