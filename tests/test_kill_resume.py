"""
Kill-and-resume integration test. Requires CoMLRL installed (runs on AMD server).
Skip locally if comlrl is not importable.
"""
import os
import signal
import subprocess
import sys
import time
import tempfile
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="Set RUN_INTEGRATION=1 to run on AMD server"
)


def test_kill_and_resume(tmp_path):
    sessions_dir = str(tmp_path / "sessions")
    ckpt_dir = str(tmp_path / "checkpoints")

    config = f"""
seed: 42
model: "Qwen/Qwen3-1.7B"
num_agents: 2
num_train_epochs: 1
reward_func: "dummy"
watchdog_shm: "magrpo_kill_resume_test"
model_params:
  enable_thinking: false
  joint_mode: "aligned"
  num_turns: 1
  num_generations: 2
  max_new_tokens: 64
  train_batch_size: 1
  rollout_buffer_size: 2
  agent_learning_rate: 1.0e-6
dataset:
  type: inline
  prompts:
    - "Write a function that adds two numbers:"
    - "Write a function that reverses a string:"
    - "Write a function that checks if a number is prime:"
    - "Write a function that computes factorial:"
checkpoint:
  output_dir: {ckpt_dir}
  save_steps: 1
reward_guard: false
"""
    cfg_path = str(tmp_path / "config.yaml")
    with open(cfg_path, "w") as f:
        f.write(config)

    log_path = str(tmp_path / "train.log")
    # Start trainer with output captured for diagnostics
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            [sys.executable, "run.py", "--config", cfg_path,
             "--sessions-dir", sessions_dir],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )

    # Wait for at least one checkpoint to appear
    deadline = time.time() + 120
    while time.time() < deadline:
        import re as _re
        ckpts = [d for d in os.listdir(ckpt_dir)
                 if _re.match(r"^ckpt-\d{6}$", d)] if os.path.isdir(ckpt_dir) else []
        if ckpts:
            break
        if proc.poll() is not None:
            break  # process exited — check log below
        time.sleep(2)

    if not ckpts and os.path.exists(log_path):
        print("\n--- subprocess log ---")
        print(open(log_path).read()[-3000:])
        print("--- end log ---")

    assert ckpts, f"No completed checkpoint within 120s (proc exit={proc.poll()})"

    # Kill mid-run
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=15)

    # Resume from latest checkpoint
    from src.training.checkpoint import CheckpointManager
    cm = CheckpointManager(ckpt_dir, keep=2)
    latest = cm.latest_path()
    assert latest is not None

    resume_proc = subprocess.run(
        [sys.executable, "run.py", "--config", cfg_path,
         "--sessions-dir", sessions_dir],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        timeout=300,
    )
    assert resume_proc.returncode == 0
