"""Deterministic generator and gate for the offline quick-start replay."""
import json


class ReplayDemoAdapter:
    def generate(self, prompt, *, max_new_tokens, schema=None):
        if schema is not None:
            return {"text": '{"questions": ["Which club does Alex Rivera play for?"]}'}
        state = json.loads(prompt.rsplit("\n", 1)[1])
        if "followup_question" in state:
            return {"text": "North Town FC"}
        if state["executed_followups"]:
            return {"text": state["executed_followups"][-1]["answer"]}
        return {"text": "Alex Rivera"}

    def predict(self, kind, request):
        choice = "sufficient" if kind == "sufficiency" else "execute"
        return {"choice": choice, "probabilities": {
            option["value"]: float(option["value"] == choice)
            for option in request["options"]
        }}


def build(config):
    return ReplayDemoAdapter()
