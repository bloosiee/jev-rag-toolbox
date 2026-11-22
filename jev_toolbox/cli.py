"""Console entry point."""
import argparse
import json
from pathlib import Path

from .adapters import load_adapter, predict as adapter_predict
from .converters import convert
from .evaluation import evaluate
from .model import DecisionModel
from .protocol import SPECS, atomic, file_sha
from .training import prepare, verify
from .rag.commands import add_parser as add_rag_parser


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def infer(args, rows, task):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty list of decision requests")
    identifiers = [row["id"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Decision request IDs must be unique")
    if args.adapter:
        adapter = load_adapter(args.adapter, args.adapter_config)
        return [adapter_predict(adapter, task, row) for row in rows]
    model = DecisionModel(args.method, args.base, args.checkpoint, args.device)
    return [model.predict(task, row) for row in rows]


def main():
    parser = argparse.ArgumentParser(prog="jev")
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict", help="Score one request")
    predict.add_argument("--task", choices=SPECS, required=True)
    predict.add_argument("--request", type=Path, required=True)
    benchmark = commands.add_parser("eval", help="Score a frozen validation or test split")
    inputs = benchmark.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--config", type=Path)
    inputs.add_argument("--requests", type=Path)
    benchmark.add_argument("--split", choices=["val", "test"])
    benchmark.add_argument("--task", choices=SPECS)
    benchmark.add_argument("--gold", type=Path)
    benchmark.add_argument("--output", type=Path, required=True)
    for command in (predict, benchmark):
        command.add_argument("--method", choices=["jev", "qwen", "rule"], default="jev")
        command.add_argument("--base", type=Path)
        command.add_argument("--checkpoint", type=Path)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--adapter", help="Local Python module:factory")
        command.add_argument("--adapter-config", type=Path)
    train = commands.add_parser("prepare-train", help="Verify inputs and create a fresh run")
    for name in ("config", "output", "base", "initial", "mentat-src"):
        train.add_argument(f"--{name}", type=Path, required=True)
    converter = commands.add_parser("convert", help="Convert JSON, JSONL or CSV records")
    converter.add_argument("kind", choices=["decisions", "rag"])
    converter.add_argument("--input", type=Path, required=True)
    converter.add_argument("--output", type=Path, required=True)
    converter.add_argument("--mapping", type=Path)
    converter.add_argument("--task", choices=SPECS)
    converter.add_argument("--labeled", action="store_true")
    add_rag_parser(commands)
    args = parser.parse_args()
    if getattr(args, "adapter_config", None) and not args.adapter:
        parser.error("--adapter-config requires --adapter")
    if getattr(args, "adapter", None):
        args.method = "adapter"
    if args.command == "convert":
        result = convert(args)
    elif args.command == "rag":
        result = args.handler(args)
    elif args.command == "predict":
        result = infer(args, [read(args.request)], args.task)[0]
    elif args.command == "eval" and args.requests:
        if not args.task or args.split:
            parser.error("--requests requires --task and takes no --split")
        args.output.mkdir(parents=True, exist_ok=False)
        predictions = infer(args, read(args.requests), args.task)
        output = args.output / "predictions.json"
        atomic(output, predictions)
        result = {"n": len(predictions), "task": args.task, "method": args.method,
                  "requests_sha256": file_sha(args.requests), "predictions_sha256": file_sha(output)}
        if args.adapter:
            result["adapter"] = args.adapter
        if args.gold:
            result.update(evaluate(args.task, predictions, read(args.gold)),
                          gold_sha256=file_sha(args.gold))
        atomic(args.output / "metrics.json", result)
    else:
        config = read(args.config)
        if args.command == "prepare-train":
            result = prepare(config, args.config.parent, args.output, args.base, args.initial, args.mentat_src)
        else:
            if not args.split or args.task or args.gold:
                parser.error("--config requires --split and takes no --task or --gold")
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
            if args.adapter:
                result["adapter"] = args.adapter
            if args.method in {"jev", "qwen"}:
                result["checkpoint_sha256"] = {
                    name: file_sha(args.checkpoint / name)
                    for name in ("adapter_config.json", "adapter_model.safetensors",
                                 "decision_readout.json", "decision_readout.safetensors")
                }
            atomic(args.output / "metrics.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
