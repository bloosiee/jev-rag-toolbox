"""Convert tabular or nested source records to toolbox inputs."""
import csv
import json

from .model import normalize
from .protocol import SPECS, atomic, file_sha


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        if path.suffix.lower() == ".csv":
            rows = list(csv.DictReader(stream))
        elif path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in stream if line.strip()]
        elif path.suffix.lower() == ".json":
            rows = json.load(stream)
        else:
            raise ValueError("Expected a .csv, .jsonl or .json file")
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Expected a nonempty collection of records")
    return rows


def field(row, name):
    if name in row:
        return row[name]
    for part in name.split("."):
        row = row[part]
    return row


def structured(value):
    return json.loads(value) if isinstance(value, str) else value


def identifier(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError("ID must be a nonempty string or integer")
    return str(value)


def text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def documents(value, decision):
    docs = structured(value)
    if not isinstance(docs, list) or (not docs and not decision):
        raise ValueError("Expected a document list")
    result = []
    for doc in docs:
        if not isinstance(doc, dict):
            raise ValueError("Each document needs title and text or sentences")
        title = text(doc["title"], "title")
        sentences = [doc["text"]] if "text" in doc else doc["sentences"]
        if (not isinstance(sentences, list) or not all(isinstance(s, str) for s in sentences)
                or not "".join(sentences).strip()):
            raise ValueError("Document content must be nonempty text or a list of strings")
        result.append({"title": title, "text": "".join(sentences)} if decision
                      else {"title": title, "sentences": sentences})
    return result


def decision_state(row, mapping, task):
    fields = {"original_question", "retrieved_documents"}
    if task == "necessity":
        fields |= {"executed_followups", "current_intermediate_answer", "pending_questions", "proposed_followup"}
    if "state" in mapping or "state" in row:
        source = structured(field(row, mapping.get("state", "state")))
        state = {key: source[key] for key in sorted(fields)}
    else:
        state = {key: field(row, mapping.get(key, key)) for key in sorted(fields)}
    state["original_question"] = text(state["original_question"], "original_question")
    state["retrieved_documents"] = documents(state["retrieved_documents"], decision=True)
    if task == "necessity":
        for key in ("executed_followups", "pending_questions"):
            state[key] = structured(state[key])
            if not isinstance(state[key], list):
                raise ValueError(f"{key} must be a list")
        state["executed_followups"] = [
            {"question": text(item["question"], "followup question"),
             "answer": text(item["answer"], "followup answer")}
            for item in state["executed_followups"]
        ]
        for item in state["pending_questions"]:
            text(item, "pending question")
        draft = state["current_intermediate_answer"]
        if draft is not None and not isinstance(draft, str):
            raise ValueError("current_intermediate_answer must be a string or null")
        text(state["proposed_followup"], "proposed_followup")
    return state


def convert(args):
    mapping = json.loads(args.mapping.read_text(encoding="utf-8")) if args.mapping else {}
    if not isinstance(mapping, dict) or any(not isinstance(value, str) for value in mapping.values()):
        raise ValueError("Mapping must be a JSON object of source field names")
    decision = args.kind == "decisions"
    if decision and not args.task:
        raise ValueError("--task is required for decision logs")
    if not decision and args.task:
        raise ValueError("--task applies only to decision logs")
    rows, gold, seen = [], {}, set()
    label_field = "label" if decision else "answer"
    for index, item in enumerate(read_rows(args.input), start=1):
        try:
            uid = identifier(field(item, mapping.get("id", "id")))
            if uid in seen:
                raise ValueError(f"Duplicate ID: {uid}")
            seen.add(uid)
            if decision:
                row = normalize(args.task, {"id": uid, "state": decision_state(item, mapping, args.task)})
            else:
                row = {"id": uid, "question": text(field(item, mapping.get("question", "question")), "question"),
                       "documents": documents(field(item, mapping.get("documents", "documents")), decision=False)}
            rows.append(row)
            if args.labeled:
                label = text(field(item, mapping.get(label_field, label_field)), label_field)
                if decision:
                    options = [choice for choice, _ in SPECS[args.task][1]]
                    if label not in options:
                        raise ValueError(f"Label must be one of {options}")
                    label = {"label": label} if args.task == "necessity" else {"sufficient": label == "sufficient"}
                gold[uid] = label
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Record {index}: {exc}") from exc
    args.output.mkdir(parents=True, exist_ok=False)
    name = "requests.json" if decision else "tasks.json"
    atomic(args.output / name, rows)
    files = [name]
    if args.labeled:
        atomic(args.output / "gold.json", gold)
        files.append("gold.json")
    atomic(args.output / "FILES_SHA256.json", {name: file_sha(args.output / name) for name in files})
    return {"kind": args.kind, "records": len(rows), "labeled": args.labeled, "output": str(args.output)}
