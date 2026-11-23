# 实验结果

按**答案生成模型**分组。`Jev v1` 和 `Jev v2` 是门控权重版本，表中无门控和 Qwen 门控是同数据对照。

## Qwen3.5-4B 生成器

### HotpotQA 400

| 实验 | 结果 | 轨迹 |
| --- | --- | --- |
| v1 主实验 | [报告](HOTPOTQA400_V1.md) · [JSON](hotpotqa400.json) | |
| v1 门控消融 | [报告](HOTPOTQA400_GATE_ABLATION.md) · [JSON](hotpotqa400_gate_ablation.json) | [轨迹索引](hotpotqa400_gate_traces.jsonl.gz) |
| v2 主实验 | [报告](HOTPOTQA400_V2.md) · [JSON](hotpotqa400_v2.json) | [轨迹索引](hotpotqa400_v2_traces.jsonl.gz) |
| v2 门控消融 | [报告](HOTPOTQA400_V2_ABLATION.md) · [JSON](hotpotqa400_v2_ablation.json) | [轨迹索引](hotpotqa400_v2_ablation_traces.jsonl.gz) |

### 2WikiMultiHopQA 400

[v1 主实验 JSON](2wikimultihopqa400.json) · [v2 主实验报告](2WIKIMULTIHOPQA400_V2.md) · [v2 轨迹](2wikimultihopqa400_v2_traces.jsonl.gz)

## Qwen3.5-9B 生成器

此分区用于独立的 9B 生成器对照。计划在固定题集上比较 **9B 无门控**和**9B + Jev v2 95/95**，两组共享 9B 初稿。目前暂无已核验结果。

## 查看报告与轨迹

用 `jev rag compare --published results/hotpotqa400.json` 查看 v1 主实验汇总。用 `jev rag trace --input <轨迹索引> --id <题号>` 查看逐步轨迹。 [命令与字段说明](../docs/EXPERIMENTS.md)
