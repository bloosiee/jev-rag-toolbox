"""Compare and replay a fictional three-hop RAG question offline."""
import argparse
from pathlib import Path
import sys
from tempfile import mkdtemp

from jev_toolbox.rag.commands import read
from jev_toolbox.rag.experiments import compare, diagnose
from jev_toolbox.rag.replay import replay
from jev_toolbox.rag.trace import branches
from jev_toolbox.rag.utils import atomic

HERE = Path(__file__).resolve().parent
FIXTURE_ID = "demo-glass-meridian"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,help="A new directory for results and traces")
    args = parser.parse_args()
    if args.output is None:
        runs = Path("runs")
        runs.mkdir(exist_ok=True)
        root = Path(mkdtemp(prefix="replay-demo-",dir=runs))
    else:
        root = args.output
        root.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(HERE/"adapters"))
    from replay_demo_adapter import COUNTRY, DOCUMENTS, PUBLICATION_QUERY, QUESTION

    tasks,gold = root/"tasks.json",root/"gold.json"
    atomic(tasks,[{"id":FIXTURE_ID,"question":QUESTION,"documents":DOCUMENTS}])
    atomic(gold,{FIXTURE_ID:COUNTRY})
    config = read(HERE.parent/"configs/hotpotqa_compare.json")
    config["pipeline"].update(initial_documents=1,documents_per_branch=2)
    preset = root/"compare-config.json"
    atomic(preset,config)
    comparison = root/"comparison"
    report = compare(argparse.Namespace(
        published=None,config=preset,tasks=tasks,gold=gold,output=comparison,
        manifest=None,base=None,checkpoint=None,adapter_config=None,
        generator_adapter="replay_demo_adapter:build_generator",
        gate_adapter="replay_demo_adapter:build_jev_gate",
        qwen_gate_adapter="replay_demo_adapter:build_qwen_gate"))
    diagnosis = diagnose(argparse.Namespace(input=[comparison/"qwen",comparison/"jev"],id=FIXTURE_ID,
                        arms="qwen,jev",output=root/"diagnosis.json"))
    steps = diagnosis["arms"][1]["steps"]
    stop = next(e for e in steps if e.get("module")=="sufficiency" and e["choice"]=="sufficient")
    skip = next(e for e in steps if e.get("module")=="necessity" and e["choice"]=="skip"
                and e["request"]["state"]["proposed_followup"]==PUBLICATION_QUERY)
    print(f"Question: {QUESTION}\n")
    print("Policy     | Answer                  | EM | Generation calls | Gate calls")
    for arm in ("fixed","qwen","jev"):
        trace = next(branches(comparison/arm,FIXTURE_ID))
        m = report["metrics"][arm]
        print(f"{arm:10} | {trace['summary']['answer']:23} | {m['EM']:.0f}  | {m['generation_calls']:16} | {m['gate_calls']}")
    print(f"\nFirst differing branch action: {diagnosis['differences'][0]['first_different_branch_action']}")
    for event,directory in ((stop,"replay-stop"),(skip,"replay-skip")):
        node = f"{event['step']:03d}_{event['module']}"
        result = replay(argparse.Namespace(run=comparison/"jev",tasks=tasks,output=root/directory,
                        question_id=FIXTURE_ID,node=node,options="all",adapter_config=None,gold=gold))
        print(f"\n{node} (recorded action: {result['original_choice']})")
        print("Forced choice | Final answer            | EM | Generation calls after fork | Gate calls after fork")
        for choice,row in result["branches"].items():
            suffix = row["suffix"]
            print(f"{choice:13} | {row['answer']:23} | {row['EM']}  | "
                  f"{suffix['generator_calls']:27} | {suffix['jev_calls']}")
    print(f"\nComparison, diagnosis, and full replay reports: {root}")


if __name__=="__main__":
    main()
