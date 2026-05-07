"""
P1b — Single-Agent Baseline Evaluator (generation only).

Loads Qwen3-1.7B, generates completions for a pre-staged HumanEval problems
JSON file, writes samples.jsonl and generation_stats.json.

The demo runner (demo/scripts/run_baseline.sh) is responsible for:
  - Staging the problems JSON into the Docker volume
  - Calling evaluate_functional_correctness on the samples file
  - Merging test_pass_rate into the final baseline_metrics.json artifact

Usage (from /workspace inside Docker):
    python -m baseline.eval_baseline \\
        --config configs/_run_config.yaml \\
        --problems-path artifacts/P1b/problems.json \\
        --samples-path artifacts/P1b/samples.jsonl \\
        --stats-path artifacts/P1b/generation_stats.json
"""
import sys
import os
import json
import time
import argparse
import datetime
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from baseline.metrics import compute_syntactic_ratio, compute_token_throughput


def assert_no_think_tokens(text: str):
    if "<think>" in text or "</think>" in text:
        raise AssertionError(f"Thinking token detected: {text[:200]!r}")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def generate_completions(
    model, tokenizer, problems: dict, max_new_tokens: int, batch_size: int = 1
) -> tuple[list[dict], int]:
    """Return (completions, total_tokens_generated).

    completions: list of {"task_id": str, "completion": str}
    total_tokens: sum of generated token counts across all problems
    """
    items = list(problems.items())
    completions = []
    total_tokens = 0
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        task_ids = [tid for tid, _ in batch]
        prompts = [problem["prompt"] for _, problem in batch]
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=False
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        for j, task_id in enumerate(task_ids):
            generated = outputs[j][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(generated, skip_special_tokens=True)
            assert_no_think_tokens(text)
            completions.append({"task_id": task_id, "completion": text})
            total_tokens += generated.shape[0]
    return completions, total_tokens


def write_samples_jsonl(completions: list[dict], path: str):
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        for item in completions:
            f.write(json.dumps(item) + "\n")
    os.replace(tmp_path, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--problems-path", required=True,
                        help="Path to staged problems JSON (dict of task_id -> problem)")
    parser.add_argument("--samples-path", default=None,
                        help="Override baseline.samples_path from config")
    parser.add_argument("--stats-path", default=None,
                        help="Override baseline.stats_path from config")
    parser.add_argument("--num-runs", type=int, default=None,
                        help="Override baseline.num_runs from config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg["seed"])

    model_name = cfg["model"]
    baseline_cfg = cfg.get("baseline", {})
    samples_path = args.samples_path or baseline_cfg.get("samples_path", "artifacts/P1b/samples.jsonl")
    stats_path = args.stats_path or baseline_cfg.get("stats_path", "artifacts/P1b/generation_stats.json")
    num_runs = args.num_runs if args.num_runs is not None else baseline_cfg.get("num_runs", 1)
    max_new_tokens = cfg["model_params"].get("max_new_tokens", 512)
    batch_size = baseline_cfg.get("batch_size", 1)

    with open(args.problems_path) as f:
        problems = json.load(f)

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.generation_config.enable_thinking = False
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    all_completions = []
    run_syntactic_ratios = []
    run_throughputs = []

    for run_idx in range(num_runs):
        print(f"Run {run_idx + 1}/{num_runs}: generating {len(problems)} completions (batch_size={batch_size})...")
        t0 = time.perf_counter()
        completions, total_tokens = generate_completions(model, tokenizer, problems, max_new_tokens, batch_size)
        elapsed = time.perf_counter() - t0

        all_completions.extend(completions)
        run_syntactic_ratios.append(compute_syntactic_ratio([c["completion"] for c in completions]))
        run_throughputs.append(compute_token_throughput(total_tokens, elapsed))

    # Combined samples file: N completions per task_id.
    # evaluate_functional_correctness uses all of them when estimating pass@1.
    write_samples_jsonl(all_completions, samples_path)

    stats = {
        "syntactic_correctness_ratio": sum(run_syntactic_ratios) / num_runs,
        "token_throughput_per_sec": sum(run_throughputs) / num_runs,
        "num_problems": len(problems),
        "num_runs": num_runs,
        "model": model_name,
        "dataset": "openai/openai_humaneval",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }

    stats_dir = os.path.dirname(stats_path)
    if stats_dir:
        os.makedirs(stats_dir, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))
    print(f"\nGeneration complete ({num_runs} run(s)). Stats written to {stats_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
