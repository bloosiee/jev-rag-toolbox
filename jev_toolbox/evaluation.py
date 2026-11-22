"""Score decision records against an independently stored answer key."""
from collections import Counter
from itertools import groupby

from .protocol import SPECS


def sufficient_auroc(pairs):
    positives = sum(truth for _, truth in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return None
    wins = 0.0
    lower_negatives = 0
    for _, tied in groupby(sorted(pairs), key=lambda pair: pair[0]):
        group = list(tied)
        group_positives = sum(truth for _, truth in group)
        group_negatives = len(group) - group_positives
        wins += group_positives * (lower_negatives + group_negatives / 2)
        lower_negatives += group_negatives
    return wins / (positives * negatives)


def evaluate(task, predictions, gold):
    labels = [value for value, _ in SPECS[task][1]]
    by_id = {row["id"]: row for row in predictions}
    if not predictions or len(by_id) != len(predictions) or set(by_id) != set(gold):
        raise ValueError("Predictions and gold must cover the same unique IDs")
    pairs = []
    for uid, row in by_id.items():
        target = gold[uid]
        label = target["label"] if task == "necessity" else (
            "sufficient" if target["sufficient"] else "insufficient"
        )
        pairs.append((label, row["choice"]))
    counts = Counter(pairs)
    correct = sum(counts[label, label] for label in labels)
    f1 = {}
    for label in labels:
        denominator = sum(counts[label, other] + counts[other, label] for other in labels)
        f1[label] = 2 * counts[label, label] / denominator if denominator else 0.0
    metrics = {
        "n": len(pairs), "correct": correct, "accuracy": correct / len(pairs),
        "macro_f1": sum(f1.values()) / len(labels), "class_f1": f1,
        "confusion": {gold: {pred: counts[gold, pred] for pred in labels} for gold in labels},
    }

    if task == "sufficiency" and all(row["probabilities"] is not None for row in predictions):
        metrics["auroc"] = sufficient_auroc([
            (row["probabilities"]["sufficient"], gold[row["id"]]["sufficient"])
            for row in predictions
        ])
    return metrics
