# HotpotQA 400 · Joint v2

The runs use the same questions, candidate documents, initial drafts, generator, and search budget. Both Jev thresholds are 95%, with binary necessity decisions. Temperatures are carried over from v1. The v1, no-gate, and Qwen-gate results are reused on the same questions. [v1 settings](../docs/ITERATIVE_RAG.md)

| Policy | EM | F1 | Qwen calls | Jev calls | Input tokens | Output tokens | Total tokens | End-to-end s/question | Qwen s/question | Jev s/question |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No gate | 52.75% | 66.46% | 3,706 | 0 | 2,400,254 | 104,250 | 2,504,504 | 14.62 | 13.93 | 0.00 |
| Qwen gate | 37.25% | 47.74% | 2,348 | 0 | 1,065,085 | 23,849 | 1,088,934 | 3.17 | 4.01 | 0.00 |
| Jev v1 | 52.25% | 65.52% | 2,602 | 1,818 | 2,758,917 | 64,446 | 2,823,363 | 10.32 | 9.02 | 1.01 |
| Jev v2 | 51.75% | 65.33% | 1,578 | 1,152 | 1,571,632 | 36,649 | 1,608,281 | 5.81 | 4.96 | 0.61 |

Compared with v1, v2 answers 6 previously incorrect questions correctly and misses 8 previously correct questions. The original runs retain their timing conventions. Cache behavior and scheduling affect end-to-end times.

## Inspect a trace

```bash
jev rag trace --input results/hotpotqa400_v2_traces.jsonl.gz \
  --id 5ab9535a5542996be202047e
```

Traces include gate actions, confidence, retrieved document indices, stop reasons, and costs. [Summary data](hotpotqa400_v2.json) · [Trace index](hotpotqa400_v2_traces.jsonl.gz)
