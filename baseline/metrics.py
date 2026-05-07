"""P1b metric computations and output schema contract."""
import ast

# Keys required by P4b dashboard — do not add or remove without updating P4b plan.
METRICS_SCHEMA = [
    "test_pass_rate",  # alias for pass@1, kept for P4b compatibility
    "pass@1",
    "pass@2",
    "pass@5",
    "syntactic_correctness_ratio",
    "token_throughput_per_sec",
    "num_problems",
    "num_runs",
    "model",
    "dataset",
    "timestamp",
]


def is_ast_parseable(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def compute_syntactic_ratio(completions: list[str]) -> float:
    if not completions:
        raise ValueError("completions list is empty")
    return sum(is_ast_parseable(c) for c in completions) / len(completions)


def compute_token_throughput(total_tokens: int, elapsed_seconds: float) -> float:
    return total_tokens / elapsed_seconds
