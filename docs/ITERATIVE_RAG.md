# DRIFT-inspired BM25 RAG

Microsoft GraphRAG's DRIFT Search uses community reports to generate an initial answer and follow-ups, then searches locally along a knowledge graph. This experiment adopts the iterative follow-up pattern and uses BM25 over each question's candidate documents. It processes follow-ups in queue order and compares three gate policies. [Microsoft Research introduction](https://www.microsoft.com/en-us/research/blog/introducing-drift-search-combining-global-and-local-search-methods-to-improve-quality-and-efficiency/) · [GraphRAG documentation](https://microsoft.github.io/graphrag/query/drift_search/)

For each question, BM25 retrieves two initial documents and Qwen writes the draft. Each round begins by checking whether the draft is complete and supported by the visible evidence. If retrieval continues, Qwen proposes one to three follow-ups. A necessity decision determines whether each follow-up runs. Execution retrieves three documents and uses Qwen to answer the follow-up. The controller adds those documents and answers to the state, then Qwen rewrites the draft. The controller returns the current draft at stopping time, with a maximum of three rounds.

| Configuration | Sufficiency | Necessity |
|---|---|---|
| `hotpotqa_pipeline_fixed.json` | Continue by round | Execute all queued follow-ups |
| `hotpotqa_pipeline_qwen.json` | Qwen selects a candidate letter | Qwen uses three options |
| `hotpotqa_pipeline_jev.json` | Jev readout, 95% threshold | Jev uses two options, 95% threshold |

The original Jev adapter and readout were jointly trained on the two tasks with `batch=16, lr=2e-5`. Both temperatures were fitted by validation NLL. Low confidence routes to continued retrieval or follow-up execution. The threshold comparison is exploratory. Qwen and Jev use different numbers of necessity options, so the table compares full policies.

## Reproduction commands

[Reproduce the HotpotQA 400 comparison](EXPERIMENTS.md#reproduce-the-hotpotqa-400-comparison) for dataset, checkpoint, evaluation, and replay commands.

`trace/` records gate states, option scores, actions, retrieved documents, and draft updates by question. `predictions.json` reports answers, tokens, calls, and time. Scoring reads the answer key separately and calculates normalized exact match and answer-token F1. The scoring code comes from HotpotQA under the included Apache 2.0 license. [Upstream scoring code](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py)

## Experiment results

[Main experiments, ablations, and traces](../results/README.md)
