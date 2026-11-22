"""Deterministic adapter for checking the input and output contract."""


class SampleAdapter:
    def __init__(self, config):
        self.answer = config.get("answer", "Ada Chen")

    def predict(self, kind, request):
        values = [option["value"] for option in request["options"]]
        choice = next((value for value in ("sufficient", "execute") if value in values), values[0])
        return {"choice": choice, "probabilities": {value: float(value == choice) for value in values}}

    def generate(self, prompt, *, max_new_tokens, schema=None):
        if schema:
            return {"text": '{"questions": ["Who designed the Cedar Bridge?"]}'}
        return {"text": self.answer}


def build(config):
    return SampleAdapter(config)
