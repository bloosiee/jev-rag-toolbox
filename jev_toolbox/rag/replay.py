"""Reconstruct the controller prefix from recorded events and execute one forced suffix."""
from time import perf_counter

from .controller import initial_answer, run_task
from .hotpot_answer_scoring import compute_exact, compute_f1
from .protocol import answer_prompt, decision_input
from .utils import atomic, digest
from ..protocol import file_sha


class RecordedPrefix:
    def __init__(self, runtime, config, source, node, choice):
        self.runtime = runtime
        self.config = config
        self.source = source
        self.node = node
        self.choice = choice
        self.started = None
        self.executed = []

    def _call(self, record, recorded, live, matches, decision=False):
        if self.started is not None:
            event = live()
            self.executed.append(event)
            return event
        event = self.source[recorded]
        if not matches(event):
            raise ValueError(f"Replay prefix mismatch at {recorded}")
        if self.node == recorded:
            if not decision:
                raise ValueError("The selected node is not a decision")
            event = dict(event, original_choice=event["choice"], choice=self.choice,
                         forced=True, selected_by="forced_replay")
            atomic(record, event)
            self.started = perf_counter()
        else:
            atomic(record, dict(event, replayed_prefix=True))
        return event

    def answer(self, state, record):
        return self._call(record, record.name, lambda: self.runtime.answer(state,record),
                          lambda event: event["prompt"] == answer_prompt(state))

    def generate_text(self, prompt, kind, limit, record, schema=None):
        return self._call(record, record.name, lambda: self.runtime.generate_text(prompt,kind,limit,record,schema),
                          lambda event: event["prompt"] == prompt and not event.get("truncated"))

    def decision(self, state, arm, record, kind):
        state_for_model, question, options = decision_input(self.config,state,kind,arm)
        def matches(event):
            request = event["request"]
            if "question" in request:
                return request == {"kind":kind,"state":state_for_model,"question":question,
                                   "options":[dict(value=v,description=d) for v,d in options]}
            return (request["state"] == state_for_model and request["arm"] == arm
                    and digest(request) == event["request_sha256"])
        return self._call(record, record.name, lambda: self.runtime.decision(state,arm,record,kind),
                          matches, decision=True)


def _usage(events):
    out = {"generator_calls":sum(e["kind"]!="decision" for e in events),
           "jev_calls":sum(e["kind"]=="decision" and e.get("arm")=="jev" for e in events),
           "generator_seconds":sum(e["wall_seconds"] for e in events if e["kind"]!="decision"),
           "jev_seconds":sum(e["wall_seconds"] for e in events if e["kind"]=="decision" and e.get("arm")=="jev")}
    for key in ("input_tokens","output_tokens"):
        values = [e.get(key) for e in events]
        out[key] = sum(values) if None not in values else None
    a,b = out["input_tokens"],out["output_tokens"]
    out["total_tokens"] = a+b if a is not None and b is not None else None
    return out


def replay(args):
    from .commands import build_runtime, read, unique_by_id

    metadata = read(args.run/"run.json")
    if "adapter_config_sha256" not in metadata:
        raise ValueError("Record a new run with this version before doing live replay")
    if metadata["tasks_sha256"] != file_sha(args.tasks):
        raise ValueError("Task file differs from the recorded run")
    if metadata["adapter_config_sha256"] != (file_sha(args.adapter_config) if args.adapter_config else None):
        raise ValueError("Adapter configuration differs from the recorded run")
    for path, expected in metadata.get("model_files_sha256", {}).items():
        if file_sha(path) != expected:
            raise ValueError(f"Model input differs from the recorded run: {path}")
    config = metadata["config"]
    if config["arm"] != "jev":
        raise ValueError("Select a recorded Jev arm")
    tasks = read(args.tasks)
    task = unique_by_id(tasks)[args.question_id]
    index = next(i for i,row in enumerate(tasks) if row["id"] == args.question_id)
    folder = args.run/"trace"/f"{index:04d}"
    arm = folder/"jev"
    original = read(arm/"result.json")
    if original["task_sha256"] != digest(task):
        raise ValueError("Original task SHA mismatch")
    recorded = {p.name:read(p) for p in arm.glob("[0-9]*.json")}
    primer = folder/"initial/initial_answer.json"
    recorded[primer.name] = read(primer)
    query = str(args.node)
    names = [name for name in recorded if (name.startswith(f"{int(query):03d}_") if query.isdecimal()
                                          else query in {name,name.removesuffix(".json")})]
    if len(names)!=1 or recorded[names[0]]["kind"]!="decision":
        raise ValueError("Node must be a recorded Jev decision step")
    node = names[0]
    target = recorded[node]
    _,_,options = decision_input(config,target["request"]["state"],target["module"],"jev")
    allowed = [value for value,_ in options]
    choices = allowed if args.options == "all" else args.options.split(",")
    if not choices or len(set(choices))!=len(choices) or not set(choices)<=set(allowed):
        raise ValueError(f"Options must be chosen from {allowed}")
    args.output.mkdir(parents=True,exist_ok=False)
    runtime = build_runtime(config,args.output,args.adapter_config)
    results = {}
    for choice in choices:
        path = args.output/choice
        adapter = RecordedPrefix(runtime,config,recorded,node,choice)
        initial = initial_answer(adapter,task,path/"initial",config)
        result = run_task(adapter,task,"jev",path/"jev",config,initial)
        if adapter.started is None:
            raise ValueError("Replay never reached the specified node")
        results[choice] = {"answer":result["answer"],"stop_reason":result["stop_reason"],
                           "retrieved_indices":result["retrieved_indices"],
                           "suffix":{"end_to_end_seconds":perf_counter()-adapter.started,**_usage(adapter.executed)}}
    if args.gold:
        gold = read(args.gold)[args.question_id]
        for row in results.values():
            row.update(EM=compute_exact(gold,row["answer"]),F1=compute_f1(gold,row["answer"]))
            row["quality_delta_vs_recorded"] = {"EM":row["EM"]-compute_exact(gold,original["answer"]),
                                                "F1":row["F1"]-compute_f1(gold,original["answer"])}
        original.update(EM=compute_exact(gold,original["answer"]),F1=compute_f1(gold,original["answer"]))
    reference = results.get(target["choice"])
    if reference is not None:
        for row in results.values():
            row["delta_vs_replayed_original_option"] = {
                k:row["suffix"][k]-value for k,value in reference["suffix"].items()
                if value is not None and row["suffix"][k] is not None}
            for key in ("EM","F1"):
                if key in row:
                    row["delta_vs_replayed_original_option"][key] = row[key]-reference[key]
    result = {"id":args.question_id,"node":node,"available_options":allowed,
              "original_choice":target["choice"],"original_recorded_answer":original["answer"],
              "original_recorded_score":{k:original[k] for k in ("EM","F1") if k in original},
              "fork_state":target["request"]["state"],
              "timing_note":"Measures only work after the forced choice; prefix and model loading excluded.",
              "branches":results}
    atomic(args.output/"replay.json",result)
    return result
