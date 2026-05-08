"""
P1b — Single-Agent Baseline Evaluator (generation + pass@k evaluation).

Instantiates one Agent, generates completions for a pre-staged HumanEval
problems JSONL file, writes samples.jsonl and generation_stats.json.

Pass@k evaluation now runs inside Docker immediately after generation when
--problems-path is supplied. The result (pass@1) is included in the stats
artifact and in the session record written to --sessions-dir (if provided).

Usage (from /workspace inside Docker):
    python -m baseline.eval_baseline \\
        --config configs/_run_config.yaml \\
        --problems-path artifacts/P1b/problems.jsonl \\
        --samples-path artifacts/P1b/samples.jsonl \\
        --stats-path artifacts/P1b/generation_stats.json \\
        --sessions-dir artifacts/P1b/sessions
"""
import sys
import os
import json
import time
import argparse
import datetime
import warnings
import yaml

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", category=UserWarning)

from baseline.agent import Agent
from baseline.metrics import compute_syntactic_ratio, compute_token_throughput
from session import Session


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


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
                        help="Path to staged problems JSONL (one problem per line)")
    parser.add_argument("--samples-path", default=None,
                        help="Override baseline.samples_path from config")
    parser.add_argument("--stats-path", default=None,
                        help="Override baseline.stats_path from config")
    parser.add_argument("--num-runs", type=int, default=None,
                        help="Override baseline.num_runs from config")
    parser.add_argument("--sessions-dir", default=None,
                        help="Directory to write session JSON (skipped if not set)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    import torch
    torch.manual_seed(cfg["seed"])

    model_name = cfg["model"]
    baseline_cfg = cfg.get("baseline", {})
    samples_path = args.samples_path or baseline_cfg.get("samples_path", "artifacts/P1b/samples.jsonl")
    stats_path = args.stats_path or baseline_cfg.get("stats_path", "artifacts/P1b/generation_stats.json")
    num_runs = args.num_runs if args.num_runs is not None else baseline_cfg.get("num_runs", 1)
    max_new_tokens = cfg["model_params"].get("max_new_tokens", 2048)
    stop_sequences = cfg["model_params"].get("stop_sequences", [])
    batch_size = baseline_cfg.get("batch_size", 1)

    problems = {}
    with open(args.problems_path) as f:
        for line in f:
            line = line.strip()
            if line:
                task = json.loads(line)
                problems[task["task_id"]] = task

    print(f"Loading agent: {model_name}")
    agent = Agent(model_name=model_name, stop_sequences=stop_sequences)

    session = Session.start({"baseline": cfg}, stage={"baseline": True, "training": False})

    all_completions = []
    run_syntactic_ratios = []
    run_throughputs = []

    for run_idx in range(num_runs):
        print(f"Run {run_idx + 1}/{num_runs}: generating {len(problems)} completions (batch_size={batch_size})...")
        t0 = time.perf_counter()
        completions, total_tokens = agent.generate_completions(problems, max_new_tokens, batch_size)
        elapsed = time.perf_counter() - t0

        all_completions.extend(completions)
        run_syntactic_ratios.append(compute_syntactic_ratio([c["completion"] for c in completions]))
        run_throughputs.append(compute_token_throughput(total_tokens, elapsed))

    write_samples_jsonl(all_completions, samples_path)

    from human_eval.evaluation import evaluate_functional_correctness
    results = evaluate_functional_correctness(
        samples_path,
        problem_file=args.problems_path,
        k=[1],
    )
    test_pass_rate = results.get("pass@1", None)

    stats = {
        "syntactic_correctness_ratio": sum(run_syntactic_ratios) / num_runs,
        "token_throughput_per_sec": sum(run_throughputs) / num_runs,
        "num_problems": len(problems),
        "num_runs": num_runs,
        "model": model_name,
        "dataset": "openai/openai_humaneval",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "test_pass_rate": test_pass_rate,
    }

    stats_dir = os.path.dirname(stats_path)
    if stats_dir:
        os.makedirs(stats_dir, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    if args.sessions_dir:
        session.update(
            {
                "test_pass_rate": test_pass_rate,
                "syntactic_correctness_ratio": stats["syntactic_correctness_ratio"],
                "token_throughput_per_sec": stats["token_throughput_per_sec"],
                "num_problems": len(problems),
                "num_runs": num_runs,
            },
            args.sessions_dir,
        )

    print(json.dumps(stats, indent=2))
    print(f"\nGeneration complete ({num_runs} run(s)). Stats written to {stats_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
