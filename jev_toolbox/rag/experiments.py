"""Matched experiments and inspection of their saved traces."""
from argparse import Namespace
from pathlib import Path

from .hotpot_answer_scoring import compute_exact

from .trace import branches
from .utils import atomic


def compare(args):
    from .commands import read, run, score_predictions

    if args.published:
        table = read(args.published)
        metrics = {}
        for name, row in table["metrics"].items():
            n = row["questions"]
            metrics[name] = {**row, "EM": row["EM_correct"]/n, "F1": row["F1_sum"]/n,
                             "total_tokens": row["input_tokens"]+row["output_tokens"]}
        result = {"recorded": True, "metrics": metrics, "paired": table["paired"]}
        if args.output:
            atomic(args.output, result)
        return result
    if args.config is None:
        raise ValueError("Live comparison requires --config")
    preset = read(args.config)
    paths = {key: getattr(args, key) for key in ("tasks", "gold", "output")}
    if not all(paths.values()):
        raise ValueError("Live comparison requires --tasks, --gold and --output")
    arms = preset["arms"]
    if not arms or len(set(arms)) != len(arms) or not set(arms) <= {"fixed", "qwen", "jev", "rule"}:
        raise ValueError("Expected a list of unique supported policies")
    args.output.mkdir(parents=True, exist_ok=False)
    metrics, outputs = {}, {}
    shared = None
    for arm in arms:
        config = {**preset["pipeline"], "arm": arm}
        directory = args.output / arm
        options = dict(vars(args), output=directory)
        options["gate_adapter"] = {"jev":args.gate_adapter, "qwen":args.qwen_gate_adapter}.get(arm)
        if not (args.generator_adapter or options["gate_adapter"]):
            options["adapter_config"] = None
        run(Namespace(**options), config=config, shared=shared)
        if shared is None:
            shared = directory
        outputs[arm] = read(directory / "predictions.json")
    # Gold is first read after all policies have finished.
    gold = read(args.gold)
    for arm, rows in outputs.items():
        metrics[arm] = score_predictions(rows, gold)
        counts = [metrics[arm].get(key) for key in ("input_tokens", "output_tokens")]
        metrics[arm]["total_tokens"] = sum(counts) if None not in counts else None
    fixed = {r["id"]: r for r in outputs.get("fixed", [])}
    paired = {}
    if fixed:
        def correct(row):
            return int(row.get("prediction_valid", True) and compute_exact(gold[row["id"]], row["answer"]))
        for arm in arms:
            if arm != "fixed":
                delta = [correct(row)-correct(fixed[row["id"]]) for row in outputs[arm]]
                paired[arm] = {"improved":delta.count(1),"regressed":delta.count(-1)}
    result = {"recorded": False, "metrics": metrics, "paired_vs_fixed": paired, "shared_initial": True,
              "timing_note": "Shared initial generation charged to each arm; model loading excluded."}
    atomic(args.output / "comparison.json", result)
    write_table(args.output / "comparison.md", metrics)
    return result


def write_table(path, metrics):
    headings = ("Policy", "Questions", "EM", "F1", "End-to-end s/question", "Generation s/question",
                "Gate s/question", "Generation calls", "Gate calls", "Input tokens", "Output tokens", "Total tokens")
    lines = ["| " + " | ".join(headings) + " |",
             "| " + " | ".join(["---"] + ["---:"]*(len(headings)-1)) + " |"]
    for arm, row in metrics.items():
        n = row["questions"]
        def average(key):
            return f"{row[key]/n:.3f}" if row.get(key) is not None else "—"
        def show(key):
            return str(row[key]) if row.get(key) is not None else "—"
        values = [arm, str(n), f"{row['EM']:.2%}", f"{row['F1']:.2%}",
                  *(average(k) for k in ("end_to_end_seconds","generation_seconds","gate_seconds")),
                  *(show(k) for k in ("generation_calls","gate_calls","input_tokens","output_tokens","total_tokens"))]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")


def _fingerprint(event):
    return tuple(event.get(key) for key in ("kind", "module", "choice", "request_sha256"))


def _actions(row):
    return [(step["round"], item.get("question"), item["action"])
            for step in row["rounds"] for item in step["branches"]]


def diagnose(args):
    rows = [row for path in args.input for row in branches(path, args.id)]
    if args.arms:
        wanted = args.arms.split(",")
        rows = [row for row in rows if row["arm"] in wanted]
        if len(rows) != len(wanted) or {r["arm"] for r in rows} != set(wanted):
            raise ValueError("Expected exactly one trajectory for each requested arm")
    if len(rows) < 2:
        raise ValueError("At least two trajectories are required")
    baseline = rows[0]
    pairs = []
    for row in rows[1:]:
        first = next((i for i,(a,b) in enumerate(zip(baseline["steps"],row["steps"]))
                      if _fingerprint(a) != _fingerprint(b)), None)
        if first is None and len(row["steps"]) != len(baseline["steps"]):
            first = min(len(row["steps"]),len(baseline["steps"]))
        old_actions, new_actions = _actions(baseline), _actions(row)
        first_action = next((i for i, (a,b) in enumerate(zip(old_actions,new_actions)) if a!=b), None)
        if first_action is None and len(old_actions)!=len(new_actions):
            first_action = min(len(old_actions),len(new_actions))
        action_delta = None
        if first_action is not None:
            action_delta = {"baseline":old_actions[first_action] if first_action<len(old_actions) else None,
                            "comparison":new_actions[first_action] if first_action<len(new_actions) else None}
        old_docs = set(baseline["summary"].get("retrieved_indices",[]))
        new_docs = set(row["summary"].get("retrieved_indices",[]))
        pairs.append({"baseline":baseline["arm"],"comparison":row["arm"],"first_different_event_index":first, "first_different_branch_action":action_delta,
                      "additional_document_indices":sorted(new_docs-old_docs),
                      "missing_document_indices":sorted(old_docs-new_docs)})
    result = {"id":args.id,"arms":rows,"differences":pairs}
    if args.output:
        atomic(args.output,result)
    return result


def add_experiment_parsers(stages):
    from .replay import replay

    p = stages.add_parser("compare", help="Run a matched multi-policy experiment or summarize a published run")
    for name in ("config","published","tasks","gold","output","manifest","base","checkpoint","adapter-config"):
        p.add_argument("--"+name,type=Path)
    for role in ("generator","gate"):
        p.add_argument(f"--{role}-adapter")
    p.add_argument("--qwen-gate-adapter", help="Optional external gate for the Qwen baseline only")
    p.set_defaults(handler=compare)
    p = stages.add_parser("diagnose", help="Compare two or more recorded trajectories for one question")
    p.add_argument("--input",type=Path,nargs="+",required=True)
    p.add_argument("--id",required=True)
    p.add_argument("--arms",help="Comma-separated names")
    p.add_argument("--output",type=Path)
    p.set_defaults(handler=diagnose)
    p = stages.add_parser("replay", help="Force a recorded Jev option and run the suffix")
    for name in ("run","tasks","output","adapter-config","gold"):
        p.add_argument("--"+name,type=Path,required=name in {"run","tasks","output"})
    p.add_argument("--question-id",required=True)
    p.add_argument("--node",required=True,help="Decision step number or recorded filename")
    p.add_argument("--options",required=True,help="Comma-separated choices or all")
    p.set_defaults(handler=replay)
