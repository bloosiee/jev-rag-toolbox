"""Console entry point."""
import argparse
import json
from pathlib import Path

from .evaluation import evaluate
from .model import DecisionModel
from .protocol import SPECS, atomic, file_sha
from .training import prepare, verify


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def infer(args, rows, task):
    model = DecisionModel(args.method, args.base, args.checkpoint, args.device)
    return [model.predict(task, row) for row in rows]


def main():
    parser = argparse.ArgumentParser(prog="jev")
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict", help="Score one request")
    predict.add_argument("--task", choices=SPECS, required=True)
    predict.add_argument("--request", type=Path, required=True)
    benchmark = commands.add_parser("eval", help="Score a frozen validation or test split")
    benchmark.add_argument("--config", type=Path, required=True)
    benchmark.add_argument("--split", choices=["val", "test"], required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    for command in (predict, benchmark):
        command.add_argument("--method", choices=["jev", "qwen", "rule"], default="jev")
        command.add_argument("--base", type=Path)
        command.add_argument("--checkpoint", type=Path)
        command.add_argument("--device", default="cuda:0")
    train = commands.add_parser("prepare-train", help="Verify inputs and create a fresh run")
    for name in ("config", "output", "base", "initial", "mentat-src"):
        train.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "predict":
        result = infer(args, [read(args.request)], args.task)[0]
    else:
        config = read(args.config)
        if args.command == "prepare-train":
            result = prepare(config, args.config.parent, args.output, args.base, args.initial, args.mentat_src)
        else:
            data = args.config.parent / config["data"]
            split = data / args.split
            manifest = read(data / "FILES_SHA256.json")
            requests_sha = f"{args.split}/requests.json"
            gold_sha = f"{args.split}/gold.json"
            verify(data, {requests_sha: manifest[requests_sha]})
            args.output.mkdir(parents=True, exist_ok=False)
            predictions = infer(args, read(split / "requests.json"), config["task"])
            output = args.output / "predictions.json"
            atomic(output, predictions)
            verify(data, {gold_sha: manifest[gold_sha]})
            result = evaluate(config["task"], predictions, read(split / "gold.json"))
            result.update(task=config["task"], split=args.split, method=args.method,
                          requests_sha256=manifest[requests_sha], gold_sha256=manifest[gold_sha],
                          predictions_sha256=file_sha(output))
            if args.method != "rule":
                result["checkpoint_sha256"] = {
                    name: file_sha(args.checkpoint / name)
                    for name in ("adapter_config.json", "adapter_model.safetensors",
                                 "decision_readout.json", "decision_readout.safetensors")
                }
            atomic(args.output / "metrics.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
