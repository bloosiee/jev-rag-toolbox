"""Look up completed trajectories by question, branch and step."""
import gzip
import json
from pathlib import Path



def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def initial_event(folder, sources):
    for local in (folder / "initial/initial_answer.json", folder / "initial.json"):
        if local.is_file():
            return read(local)
    if sources and folder.name.isdecimal():
        source = sources[int(folder.name)]
        path = Path(source["path"])
        if path.is_file():
            from ..protocol import file_sha

            if file_sha(path) != source["sha256"]:
                raise ValueError("Frozen initial record has changed")
            return read(path)["event"]
    return None


def read_branch(folder, arm, result, primer):
    files = sorted(folder.glob("[0-9]*.json"), key=lambda path: int(path.name.split("_", 1)[0]))
    steps = [{"step": 0, **primer}] if primer is not None else []
    steps.extend({"step": int(path.name.split("_", 1)[0]), **read(path)} for path in files)
    return {"id": result["id"], "arm": arm, "summary": {k:v for k,v in result.items() if k != "rounds"},
            "rounds": result.get("rounds", []), "steps": steps}


def local_branches(root, uid, arm):
    source_path = root / "INITIAL_SOURCES.json"
    sources = read(source_path) if source_path.is_file() else []
    for result_path in sorted(root.glob("trace/*/*/result.json")):
        result = read(result_path)
        if (uid is None or result["id"] == uid) and (arm is None or result_path.parent.name == arm):
            directory = result_path.parent
            yield read_branch(directory, directory.name, result, initial_event(directory.parent, sources))
    for path in sorted(root.glob("shards/*/*/outcome.json")):
        entry = read(path)
        if uid is not None and entry["id"] != uid:
            continue
        by_arm = entry.get("policies") or entry.get("arms") or ({"compact_80_80": entry["compact"]}
                   if "compact" in entry else {a: entry[a] for a in ("execute", "skip", "stop", "continue") if a in entry})
        for name, recorded in by_arm.items():
            if arm is not None and name != arm:
                continue
            base = path.parent / ("attempt" if (path.parent / "attempt").is_dir() else "")
            target = base / ("compact" if name == "compact_80_80" else name)
            target_result = target / "result.json"
            result = read(target_result) if target_result.is_file() else {**recorded, "id": entry["id"]}
            yield read_branch(target, name, result, initial_event(base, sources))


def branches(path, uid=None, arm=None):
    if path.is_dir():
        yield from local_branches(path, uid, arm)
    elif path.name.endswith((".jsonl", ".jsonl.gz")):
        stream = gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")
        with stream:
            for line in stream:
                branch = json.loads(line)
                if (uid is None or branch["id"] == uid) and (arm is None or branch["arm"] == arm):
                    yield branch
    else:
        raise ValueError("Trace input must be a run directory or a JSONL trace index")


def public_branch(branch):
    """Select trace metadata for sharing alongside reproducible results."""
    keys = ("step", "kind", "module", "choice", "raw_jev_choice", "selected_by", "confidence",
            "confidence_threshold", "probabilities", "input_tokens", "output_tokens", "wall_seconds",
            "request_sha256", "rendered_sha256", "truncated")
    steps = []
    for event in branch["steps"]:
        public = {k: event[k] for k in keys if k in event}
        if event.get("kind") == "decision" and event.get("arm") == "jev":
            public["options"] = (["sufficient", "continue"] if event["module"] == "sufficiency"
                                 else ["execute", "skip"])
        steps.append(public)
    summary_keys = ("stop_reason", "retrieved_indices", "qwen_calls", "jev_calls", "input_tokens",
                     "output_tokens", "end_to_end_seconds", "valid")
    rounds = [{"round": row["round"], "branches": [
        {key: item[key] for key in ("action", "retrieved_indices") if key in item}
        for item in row["branches"]]} for row in branch["rounds"]]
    return {"id": branch["id"], "arm": branch["arm"],
            "summary": {k:branch["summary"][k] for k in summary_keys if k in branch["summary"]},
            "rounds": rounds, "steps": steps}


def trace(args):
    records = list(branches(args.input, args.id, args.arm))
    if not records:
        raise ValueError("No matching question and branch")
    if args.id is None:
        if args.step is not None:
            raise ValueError("--step requires --id")
        return [{"id": row["id"], "arm": row["arm"], "steps": len(row["steps"])} for row in records]
    if args.step is None:
        return records
    selected = [{"id": row["id"], "arm": row["arm"], "event": step} for row in records
                for step in row["steps"] if step["step"] == args.step]
    if not selected:
        raise ValueError(f"No matching step {args.step}")
    return selected
