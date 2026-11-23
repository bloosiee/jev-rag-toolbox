# 实验结果

## HotpotQA 400

| 实验 | 结果 | 轨迹 |
| --- | --- | --- |
| v1 主实验 | [报告](HOTPOTQA400_V1.md) · [JSON](hotpotqa400.json) | |
| v1 门控消融 | [报告](HOTPOTQA400_GATE_ABLATION.md) · [JSON](hotpotqa400_gate_ablation.json) | [轨迹索引](hotpotqa400_gate_traces.jsonl.gz) |
| v2 主实验 | [报告](HOTPOTQA400_V2.md) · [JSON](hotpotqa400_v2.json) | [轨迹索引](hotpotqa400_v2_traces.jsonl.gz) |

## 2WikiMultiHopQA 400

[v1 主实验 JSON](2wikimultihopqa400.json)

用 `jev rag compare --published results/hotpotqa400.json` 查看 v1 主实验汇总。用 `jev rag trace --input <轨迹索引> --id <题号>` 查看逐步轨迹。 [命令与字段说明](../docs/EXPERIMENTS.md)
