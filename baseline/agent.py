"""Single-agent wrapper for code generation.

Owns one model + tokenizer instance. Exposes generate_completions() so the
evaluator can query one or more agents without knowing their internals.
Multi-agent benchmarking (P2a+) instantiates multiple Agent objects and
routes problems between them.
"""
import time
import torch
from transformers import AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

from utils import build_tokenizer, assert_no_think_tokens


class Agent:
    def __init__(self, model_name: str, stop_sequences: list[str] | None = None):
        self.model_name = model_name
        self.stop_sequences = stop_sequences or []

        self.tokenizer = build_tokenizer(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        self.model.generation_config.enable_thinking = False
        self.model.eval()
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

        self._stop_ids = []
        for seq in self.stop_sequences:
            ids = self.tokenizer.encode(seq, add_special_tokens=False)
            if ids:
                self._stop_ids.append(ids)

    def generate_completions(
        self,
        problems: dict,
        max_new_tokens: int,
        batch_size: int = 1,
    ) -> tuple[list[dict], int]:
        """Generate one completion per problem.

        Returns (completions, total_tokens_generated).
        completions: list of {"task_id": str, "completion": str}
        """
        stop_ids = self._stop_ids

        class StopOnSequences(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                for seq in stop_ids:
                    seq_len = len(seq)
                    if input_ids.shape[1] >= seq_len:
                        if (input_ids[:, -seq_len:] == torch.tensor(seq, device=input_ids.device)).all(dim=1).any():
                            return True
                return False

        stopping_criteria = StoppingCriteriaList([StopOnSequences()]) if stop_ids else None

        items = list(problems.items())
        completions = []
        total_tokens = 0

        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            task_ids = [tid for tid, _ in batch]
            prompts = [problem["prompt"] for _, problem in batch]

            inputs = self.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=False
            ).to(self.model.device)

            generate_kwargs = dict(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            if stopping_criteria:
                generate_kwargs["stopping_criteria"] = stopping_criteria

            with torch.no_grad():
                outputs = self.model.generate(**generate_kwargs)

            for j, task_id in enumerate(task_ids):
                generated = outputs[j][inputs["input_ids"].shape[1]:]
                text = self.tokenizer.decode(generated, skip_special_tokens=True)
                for seq in self.stop_sequences:
                    if text.endswith(seq):
                        text = text[: -len(seq)]
                assert_no_think_tokens(text, context=f"agent completion for {task_id}")
                completions.append({"task_id": task_id, "completion": text})
                total_tokens += generated.shape[0]

        return completions, total_tokens
