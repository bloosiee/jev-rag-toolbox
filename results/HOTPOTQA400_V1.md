# HotpotQA 400 · Joint v1

## Reference results

| Policy | EM | F1 | Qwen calls | Jev calls | Input tokens | Output tokens | End-to-end s/question |
|---|---:|---:|---:|---:|---:|---:|---:|
| No gate | 52.75% | 66.46% | 3,706 | 0 | 2,400,254 | 104,250 | 14.62 |
| Qwen gate | 37.25% | 47.74% | 2,348 | 0 | 1,065,085 | 23,849 | 3.17 |
| Jev 95% | 52.25% | 65.52% | 2,602 | 1,818 | 2,758,917 | 64,446 | 10.32 |

The reference combines two existing batches of 200 questions. Timings come from the original runs and depend on caching, scheduling, and hardware. Qwen calls include generation and gating. Jev calls include sufficiency and necessity judgments. Input counts reflect the tokens each model actually reads. The reproduction commands generate per-question predictions and traces. [Summary data](hotpotqa400.json)

The ablation report compares thresholds and compact sufficiency on the same questions. [Gate ablations](HOTPOTQA400_GATE_ABLATION.md) · [Rationale for gating](../docs/CONDITIONAL_INTERACTION.md)
