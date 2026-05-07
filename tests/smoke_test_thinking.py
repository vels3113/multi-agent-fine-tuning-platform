"""
Smoke test for P1a gate.
Runs run.py with the phase config and asserts no <think> tokens appear
in any generated completion.
"""
import sys
import subprocess
import os

CONFIG = "configs/p1a-thinking-gate.yaml"
LOG = "artifacts/P1a/verification-logs/smoke-test.txt"

def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)

    print(f"Running: python run.py --config {CONFIG}")
    result = subprocess.run(
        [sys.executable, "run.py", "--config", CONFIG],
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr
    with open(LOG, "w") as f:
        f.write(output)

    print(output)

    if result.returncode != 0:
        print(f"FAIL: run.py exited with code {result.returncode}")
        sys.exit(1)

    if "<think>" in output or "</think>" in output:
        print("FAIL: <think> token detected in output")
        sys.exit(1)

    print("PASS: smoke test — no thinking tokens detected")
    sys.exit(0)

if __name__ == "__main__":
    main()
