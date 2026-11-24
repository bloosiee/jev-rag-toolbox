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

[查看 HotpotQA 消融和两套数据的 95/95 对比图](../README.md#qwen35-9b)。

## 重新生成 README 图表

在仓库根目录运行 `python scripts/plot_rag_results.py`，从已核验的 JSON 生成 `results/figures/` 下的 PNG。绘图需要 Matplotlib。开展新一轮实验时，用 `--extra` 传入额外的图表定义，每个定义包含 `title`、`filename` 和 `rows`。每行依次填写显示名称、颜色键（`fixed`、`qwen` 或 Jev 版本）和该组完整的 400 题结果对象。同一图内保留同一题集和生成模型。缺少核验结果的组合标为待完成。

[查看轨迹与回放命令](../docs/EXPERIMENTS.md#查看逐步轨迹)
