"""Prepare HotpotQA requests, run without gold, then score saved outcomes."""
import json
from pathlib import Path

from .controller import initial_answer, run_task
from .hotpot_answer_scoring import compute_exact, compute_f1
from .utils import atomic, digest
from ..protocol import file_sha


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(args):
    manifest = read(args.manifest)
    parquet = args.source.suffix == ".parquet"
    if parquet:
        from pyarrow.parquet import read_table
        if file_sha(args.source) != manifest["source_parquet_sha256"]:
            raise ValueError("HotpotQA Parquet SHA-256 mismatch")
        source = read_table(args.source).to_pylist()
    else:
        source = read(args.source)
    by_id = unique_by_id(source) if parquet else unique_by_id([
        {**row, "id": row["_id"]} for row in source
    ])
    tasks, gold = [], {}
    for entry in manifest["tasks"]:
        uid = entry["id"]
        row = by_id[uid]
        context = row["context"]
        documents = zip(context["title"], context["sentences"]) if parquet else context
        task = {"id": uid, "question": row["question"], "documents": [
            {"title": title, "sentences": sentences} for title, sentences in documents
        ]}
        if digest(task) != entry["request_sha256"]:
            raise ValueError(f"Source does not match the frozen request: {uid}")
        tasks.append(task)
        gold[uid] = row["answer"]
    if digest(gold) != manifest["answer_key_sha256"]:
        raise ValueError("HotpotQA answer key SHA-256 mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in (("tasks.json", tasks), ("gold.json", gold)):
        path = args.output / name
        if path.exists() and read(path) != data:
            raise FileExistsError(path)
        atomic(path, data)
    return {"questions": len(tasks), "requests": str(args.output / "tasks.json")}


def unique_by_id(rows):
    by_id = {row["id"]: row for row in rows}
    if not by_id or len(by_id) != len(rows):
        raise ValueError("Expected nonempty rows with unique IDs")
    return by_id


def run(args):
    from .runtime import Runtime

    config = read(args.config)
    arm = config["arm"]
    if arm not in {"fixed", "qwen", "jev"}:
        raise ValueError(f"Unknown arm: {arm}")
    tasks = read(args.tasks)
    unique_by_id(tasks)
    if args.manifest:
        manifest = read(args.manifest)
        if [{"id": row["id"], "request_sha256": digest(row)} for row in tasks] != manifest["tasks"]:
            raise ValueError("Task order or input bytes differ from the frozen manifest")
        for key, expected in manifest["model_files"].items():
            role, filename = key.split("/", 1)
            directory = args.base if role == "base" else args.checkpoint
            if file_sha(directory / filename) != expected:
                raise ValueError(f"Model input hash mismatch: {key}")
    args.output.mkdir(parents=True, exist_ok=False)
    config.update(answer_model=str(args.base.resolve()), jev_model=str(args.checkpoint.resolve()))
    atomic(args.output / "run.json", {"config": config, "tasks_sha256": file_sha(args.tasks)})
    runtime = Runtime(config, args.output)
    predictions = []
    for index, task in enumerate(tasks):
        directory = args.output / "trace" / f"{index:04d}"
        initial = initial_answer(runtime, task, directory / "initial", config)
        result = run_task(runtime, task, arm, directory / arm, config, initial)
        predictions.append({k: v for k, v in result.items() if k != "rounds"})
    atomic(args.output / "predictions.json", predictions)
    return {"arm": arm, "questions": len(predictions), "output": str(args.output)}


def score_predictions(rows, gold):
    predictions = unique_by_id(rows)
    if predictions.keys() != gold.keys():
        raise ValueError("Prediction and label IDs differ")
    exact = f1 = 0
    for uid, row in predictions.items():
        if row.get("prediction_valid", True):
            exact += compute_exact(gold[uid], row["answer"])
            f1 += compute_f1(gold[uid], row["answer"])
    n = len(predictions)
    keys = ("qwen_calls", "jev_calls", "input_tokens", "output_tokens",
            "end_to_end_seconds", "qwen_seconds", "jev_seconds")
    totals = {key: sum(row[key] for row in rows) for key in keys if all(key in row for row in rows)}
    return {"questions": n, "exact": exact, "EM": exact / n, "F1": f1 / n, **totals}


def score(args):
    result = score_predictions(read(args.predictions), read(args.gold))
    atomic(args.output, result)
    return result


def add_parser(commands):
    group = commands.add_parser("rag", help="Prepare, run and score iterative RAG")
    stages = group.add_subparsers(dest="rag_command", required=True)
    specs = (
        ("prepare", prepare, ("source", "manifest", "output"), ()),
        ("run", run, ("config", "tasks", "base", "checkpoint", "output"), ("manifest",)),
        ("score", score, ("predictions", "gold", "output"), ()),
    )
    for name, handler, required, optional in specs:
        parser = stages.add_parser(name)
        for arg in required:
            parser.add_argument(f"--{arg}", type=Path, required=True)
        for arg in optional:
            parser.add_argument(f"--{arg}", type=Path)
        parser.set_defaults(handler=handler)
