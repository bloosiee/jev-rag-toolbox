"""Load local Python adapters and validate their responses."""
from importlib import import_module
import json
import math
from collections.abc import Mapping

from .model import normalize


def load_adapter(spec, config_path=None):
    module, separator, name = spec.partition(":")
    if not separator or not module or not name:
        raise ValueError("Adapter must use module:factory")
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path else {}
    if not isinstance(config, dict):
        raise ValueError("Adapter config must be a JSON object")
    return getattr(import_module(module), name)(config)


def validate_choice(result, options):
    if not isinstance(result, Mapping) or result.get("choice") not in options:
        raise ValueError(f"Adapter must return a choice in {options}")
    probabilities = result.get("probabilities")
    if probabilities is not None:
        if not isinstance(probabilities, Mapping) or set(probabilities) != set(options):
            raise ValueError("Probabilities must cover the given options exactly")
        if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p)
               or not 0 <= p <= 1 for p in probabilities.values()):
            raise ValueError("Probabilities must be finite numbers between zero and one")
        if not math.isclose(sum(probabilities.values()), 1, rel_tol=0, abs_tol=1e-6):
            raise ValueError("Probabilities must sum to one")
        if probabilities[result["choice"]] != max(probabilities.values()):
            raise ValueError("Choice must have maximum probability")
        total = sum(probabilities.values())
        probabilities = {key: value / total for key, value in probabilities.items()}
    return {"choice": result["choice"], "probabilities": probabilities}


def predict(adapter, task, item):
    row = normalize(task, item)
    options = [option["value"] for option in row["options"]]
    result = adapter.predict(task, row)
    return {"id": row["id"], "task": task, "method": "adapter",
            **validate_choice(result, options), **token_usage(result)}


def token_usage(result):
    counts = {key: result.get(key) for key in ("input_tokens", "output_tokens")}
    for key, count in counts.items():
        if count is not None and (type(count) is not int or count < 0):
            raise ValueError(f"{key} must be a nonnegative integer or null")
    return counts
