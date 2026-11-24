# Jev v3 experiment results

Each comparison uses 400 matched questions and the no-gate and Qwen-gate baselines for the same answer generator. The Jev gate uses the Qwen3.5-4B checkpoint in both settings. Thresholds are listed as sufficiency / necessity.

| Answer generator | Dataset | Results | Public trace index |
| --- | --- | --- | --- |
| Qwen3.5-4B | HotpotQA | [95/95 comparison](figures/hotpotqa-qwen4b-9595.png) · [Ablations](figures/hotpotqa-qwen4b-v3-ablation.png) | [Trace index](hotpotqa400_qwen4b_v3_traces.jsonl.gz) |
| Qwen3.5-4B | 2WikiMultiHopQA | [95/95 comparison](figures/2wiki-qwen4b-9595.png) | [Trace index](2wikimultihopqa400_qwen4b_v3_traces.jsonl.gz) |
| Qwen3.5-9B | HotpotQA | [95/95 comparison](figures/hotpotqa-qwen9b-9595.png) · [Ablations](figures/hotpotqa-qwen9b-v3-ablation.png) | [Trace index](hotpotqa400_qwen9b_v3_traces.jsonl.gz) |
| Qwen3.5-9B | 2WikiMultiHopQA | [95/95 comparison](figures/2wiki-qwen9b-9595.png) | [Trace index](2wikimultihopqa400_qwen9b_v3_traces.jsonl.gz) |

[Machine-readable figure data](jev_v3_figure_data.json) includes EM counts, F1, token counts, calls, and times. HotpotQA ablations cover 95/95 full, 95/80 full, 80/80 full, and 80/80 compact sufficiency. Capped outputs count as failures and remain in the denominator and cost totals. The public trace indexes cover successful predictions. Each 2Wiki index and each 9B HotpotQA policy has 399 traces. The 4B HotpotQA index has all 400 traces per policy.

## Main results at 95/95

| Generator | Dataset | No-gate EM | Qwen-gate EM | v1 EM | v2 EM | v3 EM | v3 F1 | v3 Qwen calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B | HotpotQA | 52.75% | 37.25% | 52.25% | 51.75% | 53.75% | 67.83% | 2,257 |
| Qwen3.5-4B | 2WikiMultiHopQA | 52.00% | 37.25% | 52.00% | 52.25% | 51.50% | 57.87% | 3,010 |
| Qwen3.5-9B | HotpotQA | 57.50% | 46.25% | 55.25% | 54.50% | 57.25% | 71.39% | 2,459 |
| Qwen3.5-9B | 2WikiMultiHopQA | 58.75% | 30.25% | 56.00% | 56.50% | 57.25% | 64.05% | 3,205 |

```bash
jev rag trace --input results/hotpotqa400_qwen4b_v3_traces.jsonl.gz --arm 95_95
jev rag diagnose --input results/hotpotqa400_qwen4b_v3_traces.jsonl.gz \
  --id QUESTION_ID --arms 95_95,80_80
```

[Download weights](../models/README.md#download-v3-weights) · [Training method](../docs/TRAINING.md#v3-supervision-from-paired-rollouts)
