import json, tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from jev_toolbox.rag.experiments import compare, diagnose
from jev_toolbox.rag.replay import replay
from jev_toolbox.rag.adapted_runtime import AdaptedRuntime

class Generator:
    def generate(self, prompt, *, max_new_tokens, schema=None):
        if schema:
            return {"text":'{"questions":["Who built it?"]}',"input_tokens":8,"output_tokens":4}
        return {"text":"Ada Chen" if "Ada Chen" in prompt else "insufficient evidence",
                "input_tokens":10,"output_tokens":1}
    def predict(self, kind, request):
        values=[row["value"] for row in request["options"]]
        choice="continue" if kind=="sufficiency" else "skip"
        return {"choice":choice,"probabilities":{v:float(v==choice) for v in values}}

with tempfile.TemporaryDirectory() as cwd:
    root=Path(cwd)
    preset=json.loads(Path("configs/hotpotqa_compare.json").read_text())
    preset["arms"].append("rule")
    preset["pipeline"].update(max_rounds=1, initial_documents=1, documents_per_branch=2)
    conf=root/"preset.json";conf.write_text(json.dumps(preset))
    task=[{"id":"q1","question":"What was the name of the bridge architect?","documents":[
        {"title":"Bridge architect","sentences":["The bridge opened in 1995."]},
        {"title":"Who built it?","sentences":["Ada Chen designed it."]}]}]
    tasks=root/"tasks.json";tasks.write_text(json.dumps(task))
    gold=root/"gold.json";gold.write_text('{"q1":"Ada Chen"}')
    out=root/"compare"
    def build(config, output, adapter_config):
        mock=Generator()
        if config["arm"]=="qwen":
            original=mock.predict
            mock.predict=lambda kind,request: original(kind,request) if kind=="sufficiency" else                 {"choice":"execute","probabilities":{o["value"]:float(o["value"]=="execute") for o in request["options"]}}
        return AdaptedRuntime(config,None,mock,mock)
    with patch("jev_toolbox.rag.commands.build_runtime",side_effect=build):
        result=compare(Namespace(config=conf,published=None,tasks=tasks,gold=gold,output=out,manifest=None,
                                 base=None,checkpoint=None,adapter_config=None,generator_adapter="mock:build",gate_adapter="mock:build",
                                 qwen_gate_adapter="mock:qwen"))
        assert result["metrics"]["jev"]["EM"]==0 and result["metrics"]["fixed"]["EM"]==1
        assert result["metrics"]["rule"]["gate_calls"]==0
        for arm in preset["arms"]:
            assert (out/arm/"trace/0000/initial/initial_answer.json").read_bytes()==(
                out/"fixed/trace/0000/initial/initial_answer.json").read_bytes()
        result=replay(Namespace(run=out/"jev",tasks=tasks,output=root/"replay",adapter_config=None,gold=gold,
                                question_id="q1",node=3,options="all"))
        assert result["branches"]["execute"]["EM"]==1 and result["branches"]["skip"]["EM"]==0
        assert result["branches"]["execute"]["suffix"]["generator_calls"]==2
        assert result["branches"]["execute"]["delta_vs_replayed_original_option"]["EM"]==1
        stop=replay(Namespace(run=out/"jev",tasks=tasks,output=root/"stop",adapter_config=None,gold=gold,
                              question_id="q1",node=1,options="all"))
        assert stop["branches"]["sufficient"]["suffix"]["generator_calls"]==0
        assert diagnose(Namespace(input=[out/"fixed",out/"jev"],id="q1",arms="fixed,jev",output=None))["differences"]
print("Four-arm compare, shared initial state, diagnosis, and both replay node types PASS")
