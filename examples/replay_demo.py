"""Record and replay the same decision with two options and no model service."""
import argparse
from pathlib import Path
import sys
from tempfile import mkdtemp

from jev_toolbox.rag.commands import read, run
from jev_toolbox.rag.replay import replay
from jev_toolbox.rag.utils import atomic

HERE = Path(__file__).resolve().parent
FIXTURE_ID = "demo-football-club"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="A new directory for the demo traces")
    args = parser.parse_args()
    if args.output is None:
        runs = Path("runs")
        runs.mkdir(exist_ok=True)
        root = Path(mkdtemp(prefix="replay-demo-", dir=runs))
    else:
        root = args.output
        root.mkdir(parents=True, exist_ok=False)
    tasks, gold = root / "tasks.json", root / "gold.json"
    atomic(tasks, [{"id": FIXTURE_ID, "question": "Which club does Alex Rivera play for?",
                    "documents": [{"title": "Player profile",
                                   "sentences": ["Alex Rivera plays for North Town FC."]}]}])
    atomic(gold, {FIXTURE_ID: "North Town FC"})
    cfg = read(HERE.parent / "configs/hotpotqa_pipeline_jev.json")
    cfg.update(max_rounds=1, initial_documents=1, documents_per_branch=1)
    sys.path.insert(0, str(HERE / "adapters"))
    backend = "replay_demo_adapter:build"
    run(argparse.Namespace(tasks=tasks, output=root / "original", manifest=None,
                           generator_adapter=backend, gate_adapter=backend, adapter_config=None,
                           base=None, checkpoint=None), config=cfg)
    report = replay(argparse.Namespace(run=root / "original", tasks=tasks, output=root / "branches",
                                       question_id=FIXTURE_ID, node="001_sufficiency", options="all",
                                       adapter_config=None, gold=gold))
    print("Forced choice | Final answer  | EM | Additional generation calls")
    for choice, branch in report["branches"].items():
        print(f"{choice:13} | {branch['answer']:13} | {branch['EM']}  | "
              f"{branch['suffix']['generator_calls']}")
    print(f"Full replay report: {root / 'branches/replay.json'}")


if __name__ == "__main__":
    main()
