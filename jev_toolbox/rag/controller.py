"""Run the full-document iterative RAG controller with interchangeable gates."""
from time import perf_counter

from .protocol import (
    NEXT_SCHEMA, branch_prompt, digest, documents, enqueue, next_prompt,
    parse_generated, public_state, retrieved,
)
from .utils import atomic


def initial_answer(runtime, task, directory, config):
    indices, _ = retrieved(task, task["question"], config["initial_documents"])
    state = public_state(task, indices, [], None, [])
    event = runtime.answer(state, directory / "initial_answer.json")
    if event["truncated"]:
        raise ValueError("Initial answer hit the output cap")
    return indices, event


def run_task(runtime, task, arm, directory, config, initial):
    directory.mkdir(parents=True, exist_ok=False)
    initial_indices, event = initial
    indices = list(initial_indices)
    draft = event["answer"]
    events = [event]
    steps = []
    history, seen, pending = [], [], []
    started = perf_counter()
    stop = "round_budget"

    def record(kind):
        return directory / f"{len(events):03d}_{kind}.json"

    def decide(state, kind):
        event = runtime.decision(state, arm, record(kind), kind)
        events.append(event)
        return event["choice"]

    for round_number in range(1, config["max_rounds"] + 1):
        state = public_state(task, indices, history, draft, pending)
        if arm != "fixed" and decide(state, "sufficiency") == "sufficient":
            stop = "sufficient"
            break

        event = runtime.generate_text(next_prompt(state, seen), "next_questions",
                                      config["max_next_tokens"], record("next_questions"),
                                      schema=NEXT_SCHEMA)
        events.append(event)
        proposed = parse_generated(event["raw"])["questions"]
        frontier = enqueue(pending, seen, proposed)
        if not pending:
            stop = "no_pending_questions"
            steps.append({"round": round_number, "frontier": frontier, "branches": []})
            break
        selected = pending[:config["max_selected_questions_per_round"]]
        del pending[:len(selected)]
        branches = []
        for index, question in enumerate(selected):
            state = public_state(task, indices, history, draft, selected[index + 1:] + pending, question)
            action = "execute" if arm == "fixed" else decide(state, "necessity")
            entry = {"question": question, "state": state, "action": action}
            if action == "execute":
                found, _ = retrieved(task, question, config["documents_per_branch"])
                context = {"original_question": task["question"], "followup_question": question,
                           "retrieved_documents": documents(task, found)}
                event = runtime.generate_text(branch_prompt(context), "branch_answer",
                                              config["max_branch_tokens"], record("branch_answer"))
                events.append(event)
                branch_answer = event["raw"].strip()
                history.append({"question": question, "answer": branch_answer})
                indices.extend(i for i in found if i not in indices)
                entry.update(retrieved_indices=found, answer=branch_answer)
            branches.append(entry)
        state = public_state(task, indices, history, draft, pending)
        event = runtime.answer(state, record("round_answer"))
        if event["truncated"]:
            raise ValueError("Round answer hit the output cap")
        events.append(event)
        draft = event["answer"]
        steps.append({"round": round_number, "frontier": frontier, "branches": branches, "draft": draft})

    own_seconds = perf_counter() - started
    adapter_events = [e for e in events if e.get("model_backend") == "adapter"]
    native_events = [e for e in events if e.get("model_backend") != "adapter"]
    jev_events = [e for e in native_events if e.get("arm") == "jev"]
    qwen_events = [e for e in native_events if e.get("arm") != "jev"]
    token_totals = {}
    for key in ("input_tokens", "output_tokens"):
        values = [e[key] for e in events]
        token_totals[key] = None if None in values else sum(values)
    result = {
        "id": task["id"], "task_sha256": digest(task), "arm": arm, "answer": draft,
        "stop_reason": stop, "rounds": steps, "retrieved_indices": indices,
        "qwen_calls": len(qwen_events), "jev_calls": len(jev_events),
        "adapter_calls": len(adapter_events),
        "adapter_seconds": sum(e["wall_seconds"] for e in adapter_events),
        **token_totals,
        "qwen_seconds": sum(e["wall_seconds"] for e in qwen_events),
        "jev_seconds": sum(e["wall_seconds"] for e in jev_events),
        "end_to_end_seconds": own_seconds + events[0]["wall_seconds"],
        "shared_initial_seconds": events[0]["wall_seconds"],
    }
    atomic(directory / "result.json", result)
    return result
