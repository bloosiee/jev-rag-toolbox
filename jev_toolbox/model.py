"""Native choice inference shared by Jev and the Qwen option-token baseline."""
from pathlib import Path
from time import perf_counter

from .protocol import SPECS, compile_prompt, digest


def normalize(task, row):
    question, options = SPECS[task]
    canonical = {
        "kind": task, "question": question,
        "options": [dict(value=value, description=desc) for value, desc in options],
    }
    for key, value in canonical.items():
        if key in row and row[key] != value:
            raise ValueError(f"{row['id']}: unexpected {key} for {task}")
    state = row["state"]
    required = {"original_question", "retrieved_documents"}
    if task == "necessity":
        required.update({"executed_followups", "current_intermediate_answer", "pending_questions", "proposed_followup"})
    if set(state) != required:
        raise ValueError(f"{row['id']}: state keys must be {sorted(required)}")
    return {"id": row["id"], "state": state, **canonical}


class DecisionModel:
    def __init__(self, method, base=None, checkpoint=None, device="cuda:0"):
        self.method = method
        if method == "rule":
            return
        if not base or not checkpoint:
            raise ValueError("Jev and Qwen require --base and --checkpoint")
        import torch
        from mentat_sys1.inference.backend import TorchDecisionEngine

        self.torch = torch
        self.device = device
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        self.engine = TorchDecisionEngine.load(
            base_model_path=Path(base), model_dir=Path(checkpoint), device=device, max_length=16384
        )

    def predict(self, task, item):
        row = normalize(task, item)
        values = [option["value"] for option in row["options"]]
        prediction = {"id": row["id"], "task": task, "method": self.method}
        if self.method == "rule":
            choice = {"sufficiency": "insufficient", "necessity": "execute"}[task]
            return dict(prediction, choice=choice, probabilities=None)
        torch = self.torch
        engine = self.engine
        prompt = compile_prompt(row)
        start = perf_counter()
        labels = [chr(65 + i) for i in range(len(values) + 1)]
        rendered, native_inputs, token_ids = engine._prepare([], prompt, labels)
        inputs = {key: value.to(self.device) for key, value in native_inputs.items()}
        base = engine.model.get_base_model()
        with torch.inference_mode():
            if self.method == "jev":
                hidden = base.model(**inputs, use_cache=False).last_hidden_state[0, -1]
                logits = engine.readout(hidden.float())[:len(values)]
            else:
                with engine.model.disable_adapter():
                    hidden = base.model(**inputs, use_cache=False).last_hidden_state[0, -1]
                    logits = base.get_output_embeddings()(hidden)[token_ids[:len(values)]]
            probabilities = torch.softmax(logits.double(), dim=0).cpu().tolist()
        return dict(
            prediction, choice=values[max(range(len(values)), key=probabilities.__getitem__)],
            probabilities=dict(zip(values, probabilities)), input_tokens=inputs["input_ids"].shape[1],
            decision_seconds=perf_counter() - start, prompt_sha256=digest(prompt),
            rendered_sha256=digest(rendered),
        )
