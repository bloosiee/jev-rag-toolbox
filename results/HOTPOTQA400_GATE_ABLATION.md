# HotpotQA 400 gate ablations

All policies use the same 400 HotpotQA distractor dev questions and DRIFT-inspired BM25 RAG. The three ablation policies share the original draft, candidate documents, and three-round limit. Published baselines were checked by question before reuse. [Method and data sources](../docs/ITERATIVE_RAG.md)

| Policy | EM | F1 | Qwen calls | Jev calls | Input tokens | Output tokens | Total tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No gate | 211/400 (52.75%) | 66.46% | 3,706 | 0 | 2,400,254 | 104,250 | 2,504,504 |
| Qwen gate | 149/400 (37.25%) | 47.74% | 2,348 | 0 | 1,065,085 | 23,849 | 1,088,934 |
| Full input Jev 95/95 | 209/400 (52.25%) | 65.52% | 2,602 | 1,818 | 2,758,917 | 64,446 | 2,823,363 |
| Full input Jev 95/80 | 206/400 (51.50%) | 63.83% | 2,292 | 1,709 | 2,290,858 | 52,053 | 2,342,911 |
| Full input Jev 80/80 | 202/400 (50.50%) | 62.83% | 1,594 | 1,199 | 1,581,653 | 34,960 | 1,616,613 |
| Compact sufficiency Jev 80/80 | 206/400 (51.50%) | 63.83% | 2,523 | 1,894 | 2,139,204 | 57,476 | 2,196,680 |

| Policy | Qwen s/question | Jev s/question | End-to-end s/question |
| --- | ---: | ---: | ---: |
| No gate | 13.93 | 0.00 | 14.62 |
| Qwen gate | 4.01 | 0.00 | 3.17 |
| Full input Jev 95/95 | 9.02 | 1.01 | 10.32 |
| Full input Jev 95/80 | 7.71 | 0.91 | 8.98 |
| Full input Jev 80/80 | 4.96 | 0.63 | 5.83 |
| Compact sufficiency Jev 80/80 | 8.65 | 0.76 | 9.82 |

Thresholds are ordered as sufficiency / necessity. The compact sufficiency prompt uses the original question, executed follow-ups and their answers, and the current draft. Necessity keeps the full input. End-to-end time includes the charged cost of the shared initial draft. Historical baselines retain their original accounting and depend on caching and scheduling.

These 400 questions were previously used for comparison. These results explore threshold and input choices. [Rationale for gating](../docs/CONDITIONAL_INTERACTION.md) · [Tracing and reproduction](../docs/EXPERIMENTS.md#reproduce-the-hotpotqa-gate-ablations)
