"""Shared utilities for training (run.py) and evaluation (baseline/eval_baseline.py)."""
from transformers import AutoTokenizer


def build_tokenizer(model_name: str, padding_side: str = "left"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side=padding_side)
    tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "eos_token_id"):
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def assert_no_think_tokens(text: str, context: str = ""):
    if "<think>" in text or "</think>" in text:
        raise AssertionError(
            f"Thinking token detected{' in ' + context if context else ''}: {text[:200]!r}"
        )
