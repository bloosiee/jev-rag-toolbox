"""Plug a generator or gate into the existing RAG controller."""
from collections.abc import Mapping
from time import perf_counter

from .confidence_policy import route
from .protocol import answer_prompt, decision_input, gate_threshold
from .utils import atomic
from ..adapters import token_usage, validate_choice


class AdaptedRuntime:
    def __init__(self, config, native, generator=None, gate=None):
        self.config = config
        self.native = native
        self.generator = generator
        self.gate = gate

    def generate_text(self, prompt, kind, limit, record, schema=None):
        if self.generator is None:
            return self.native.generate_text(prompt, kind, limit, record, schema=schema)
        return self._generate(prompt, kind, limit, record, schema)

    def answer(self, state, record):
        if self.generator is None:
            return self.native.answer(state, record)
        return self._generate(answer_prompt(state), "answer", self.config["max_answer_tokens"], record)

    def _generate(self, prompt, kind, limit, record, schema=None):
        started = perf_counter()
        result = self.generator.generate(prompt, max_new_tokens=limit, schema=schema)
        elapsed = perf_counter() - started
        if not isinstance(result, Mapping):
            raise ValueError("Generator must return a mapping")
        raw = result.get("text")
        truncated = result.get("truncated", False)
        if not isinstance(raw, str) or type(truncated) is not bool:
            raise ValueError("Generator must return text and an optional boolean truncated")
        event = {"kind": kind, "model_backend": "adapter", "prompt": prompt, "raw": raw,
                 "answer": raw.strip(), "truncated": truncated, "wall_seconds": elapsed,
                 **token_usage(result)}
        atomic(record, event)
        if truncated:
            raise ValueError(f"{kind} exceeded the generator output budget")
        return event

    def decision(self, state, arm, record, kind):
        if self.gate is None:
            return self.native.decision(state, arm, record, kind)
        state, question, options = decision_input(self.config, state, kind, arm)
        request = {"kind": kind, "state": state, "question": question,
                   "options": [dict(value=value, description=desc) for value, desc in options]}
        started = perf_counter()
        result = self.gate.predict(kind, request)
        elapsed = perf_counter() - started
        values = [value for value, _ in options]
        event = {"kind": "decision", "model_backend": "adapter", "arm": arm,
                 "module": kind, "request": request, "wall_seconds": elapsed,
                 **validate_choice(result, values), **token_usage(result)}
        if arm == "jev":
            if event["probabilities"] is None:
                raise ValueError("The Jev routing policy requires gate probabilities")
            threshold = gate_threshold(self.config, kind)
            event.update(route(kind, [event["probabilities"][key] for key in values], values,
                               threshold, self.config["low_confidence_policy"]))
            event["confidence_threshold"] = threshold
            if event["choice"] is None:
                raise ValueError("The adapter runtime requires an explicit routing action")
        atomic(record, event)
        return event
