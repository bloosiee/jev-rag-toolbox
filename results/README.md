# Experiment results

Results are grouped by the **answer generator**. Jev v1, v2, and v3 identify gate checkpoints. Each comparison uses its own no-gate and Qwen-gate baselines on the same data.

## Qwen3.5-4B generator

### HotpotQA 400

| Experiment | Results | Traces |
| --- | --- | --- |
| v1 main experiment | [Report](HOTPOTQA400_V1.md) · [JSON](hotpotqa400.json) | |
| v1 gate ablations | [Report](HOTPOTQA400_GATE_ABLATION.md) · [JSON](hotpotqa400_gate_ablation.json) | [Trace index](hotpotqa400_gate_traces.jsonl.gz) |
| v2 main experiment | [Report](HOTPOTQA400_V2.md) · [JSON](hotpotqa400_v2.json) | [Trace index](hotpotqa400_v2_traces.jsonl.gz) |
| v2 gate ablations | [Report](HOTPOTQA400_V2_ABLATION.md) · [JSON](hotpotqa400_v2_ablation.json) | [Trace index](hotpotqa400_v2_ablation_traces.jsonl.gz) |
| v3 main experiment and ablations | [Results](JEV_V3.md) | [Trace index](hotpotqa400_qwen4b_v3_traces.jsonl.gz) |

### 2WikiMultiHopQA 400

[v1 main experiment JSON](2wikimultihopqa400.json) · [v2 main experiment report](2WIKIMULTIHOPQA400_V2.md) · [v2 traces](2wikimultihopqa400_v2_traces.jsonl.gz) · [v3 results and traces](JEV_V3.md)

## Qwen3.5-9B generator

[HotpotQA ablations and 95/95 figures for both datasets](../README.md#qwen35-9b) · [v3 results and traces](JEV_V3.md)

## Regenerate the README figures

Run `python scripts/plot_rag_results.py` from the repository root to render PNGs under `results/figures/` from the checked result JSON, using Matplotlib. Use `--extra` to supply additional figure definitions. Each definition contains `title`, `filename`, and `rows`. Each row specifies a display name, a color key (`fixed`, `qwen`, or the Jev version), and the result object for a full 400-question run. Keep the dataset and generator fixed within each figure. Mark unchecked runs as pending.

[Trace and replay commands](../docs/EXPERIMENTS.md#inspect-step-by-step-traces)
