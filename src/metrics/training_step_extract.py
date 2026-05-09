"""Extract optional evaluation-facing scalars from trainer payloads (best-effort, no fabrication)."""


def infer_task_pass_rate(completions_data) -> float | None:
    """Return task pass rate in **percent** (0–100) if present on structured payloads.

    Supported shapes:
    - dict with ``task_pass_rate``, ``test_pass_rate``, or ``pass_rate`` (number or numeric string).
    - dict with ``eval`` sub-dict carrying those keys.
    - iterable of dict items with ``passed`` / ``correct`` booleans → mean * 100.
    """
    if completions_data is None:
        return None

    if isinstance(completions_data, dict):
        for key in ("task_pass_rate", "test_pass_rate", "pass_rate"):
            if key in completions_data and completions_data[key] is not None:
                try:
                    v = float(completions_data[key])
                    if 0.0 <= v <= 1.0:
                        return round(v * 100.0, 4)
                    return v
                except (TypeError, ValueError):
                    pass
        ev = completions_data.get("eval")
        if isinstance(ev, dict):
            r = infer_task_pass_rate(ev)
            if r is not None:
                return r

    if isinstance(completions_data, (list, tuple)):
        flags = []
        for item in completions_data:
            if isinstance(item, dict):
                for fk in ("passed", "correct", "success"):
                    if fk in item and isinstance(item[fk], (bool, int)):
                        flags.append(bool(item[fk]))
        if flags:
            return round(100.0 * sum(1 for f in flags if f) / len(flags), 4)

    return None


def infer_syntactic_correctness_ratio(completions_data) -> float | None:
    """Return ratio in [0, 1] when structured payloads expose it."""
    if completions_data is None or not isinstance(completions_data, dict):
        return None
    for key in ("syntactic_correctness_ratio", "syntactic_correctness", "syntax_ok_ratio"):
        if key in completions_data and completions_data[key] is not None:
            try:
                v = float(completions_data[key])
                return v if v <= 1.000001 else v / 100.0
            except (TypeError, ValueError):
                pass
    ev = completions_data.get("eval")
    if isinstance(ev, dict):
        return infer_syntactic_correctness_ratio(ev)
    return None
