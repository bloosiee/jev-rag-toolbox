# HotpotQA 400 · v2 gate ablations

The runs use the same 400 questions, original drafts, candidate documents, budget, and v1 temperatures. Each v2 setting is compared with its v1 counterpart. Cache behavior and scheduling affect historical timings. [v1 ablations](HOTPOTQA400_GATE_ABLATION.md) · [v2 main experiment](HOTPOTQA400_V2.md)

| Policy | EM | F1 | Qwen calls | Jev calls | Input tokens | Output tokens | Total tokens | End-to-end s/question | Qwen s/question | Jev s/question |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No gate | 211/400 (52.75%) | 66.46% | 3,706 | 0 | 2,400,254 | 104,250 | 2,504,504 | 14.62 | 13.93 | 0.00 |
| Qwen gate | 149/400 (37.25%) | 47.74% | 2,348 | 0 | 1,065,085 | 23,849 | 1,088,934 | 3.17 | 4.01 | 0.00 |
| v1 Full input 95/95 | 209/400 (52.25%) | 65.52% | 2,602 | 1,818 | 2,758,917 | 64,446 | 2,823,363 | 10.32 | 9.02 | 1.01 |
| v2 Full input 95/95 | 207/400 (51.75%) | 65.33% | 1,578 | 1,152 | 1,571,632 | 36,649 | 1,608,281 | 5.81 | 4.96 | 0.61 |
| v1 Full input 95/80 | 206/400 (51.50%) | 63.83% | 2,292 | 1,709 | 2,290,858 | 52,053 | 2,342,911 | 8.98 | 7.71 | 0.91 |
| v2 Full input 95/80 | 201/400 (50.25%) | 63.18% | 1,572 | 1,182 | 1,528,608 | 34,969 | 1,563,577 | 5.73 | 4.88 | 0.61 |
| v1 Full input 80/80 | 202/400 (50.50%) | 62.83% | 1,594 | 1,199 | 1,581,653 | 34,960 | 1,616,613 | 5.83 | 4.96 | 0.63 |
| v2 Full input 80/80 | 204/400 (51.00%) | 63.73% | 1,357 | 1,027 | 1,319,124 | 29,690 | 1,348,814 | 4.77 | 4.05 | 0.53 |
| v1 Compact sufficiency 80/80 | 206/400 (51.50%) | 63.83% | 2,523 | 1,894 | 2,139,204 | 57,476 | 2,196,680 | 9.82 | 8.65 | 0.76 |
| v2 Compact sufficiency 80/80 | 203/400 (50.75%) | 63.25% | 2,371 | 1,774 | 1,967,748 | 54,699 | 2,022,447 | 9.02 | 7.96 | 0.69 |

| v2 vs. matching v1 setting | Wrong to correct | Correct to wrong |
|---|---:|---:|
| Full input 95/80 | 3 | 8 |
| Full input 80/80 | 7 | 5 |
| Compact sufficiency 80/80 | 4 | 7 |

[Summary data](hotpotqa400_v2_ablation.json) · [Step-by-step traces](hotpotqa400_v2_ablation_traces.jsonl.gz)
