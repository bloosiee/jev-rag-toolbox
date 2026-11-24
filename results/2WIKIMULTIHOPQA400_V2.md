# 2WikiMultiHopQA 400 · Joint v2

All four policies use the same questions, candidate documents, and initial drafts. Jev v2 uses full-input 95/95 and the v1 temperatures and search budget. The three historical references retain their original timing conventions.

| Policy | EM | F1 | Qwen calls | Jev calls | Input tokens | Output tokens | Total tokens | End-to-end s/question | Qwen s/question | Jev s/question |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No gate | 208/400 (52.00%) | 58.33% | 3,760 | 0 | 2,580,569 | 99,698 | 2,680,267 | 14.18 | 13.51 | 0.00 |
| Qwen gate | 149/400 (37.25%) | 42.08% | 4,377 | 0 | 1,897,151 | 55,478 | 1,952,629 | 8.96 | 8.69 | 0.00 |
| Jev v1 95/95 | 208/400 (52.00%) | 58.57% | 2,907 | 1,990 | 3,353,040 | 68,967 | 3,422,007 | 11.26 | 9.75 | 1.20 |
| Jev v2 95/95 | 209/400 (52.25%) | 59.22% | 2,157 | 1,509 | 2,479,487 | 51,113 | 2,530,600 | 8.39 | 7.15 | 0.91 |

Compared with v1, v2 answers 15 previously incorrect questions correctly and misses 14 previously correct questions. One v2 output hit the generation cap and counts as incorrect in the 400-question denominator.

[Summary data](2wikimultihopqa400_v2.json) · [Step-by-step traces](2wikimultihopqa400_v2_traces.jsonl.gz)
