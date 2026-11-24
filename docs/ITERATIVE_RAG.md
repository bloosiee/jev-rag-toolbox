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

Install the model dependencies and set `MENTAT_SRC`, `BASE`, and `JOINT_CKPT`. The HotpotQA split manifest records the joint checkpoint and base checksums. Use the pinned distractor dev Parquet or the official distractor dev JSON. Preparation verifies the requests and answer key by question. [HotpotQA data and paper](https://hotpotqa.github.io/) · [Pinned dataset revision](https://huggingface.co/datasets/hotpotqa/hotpot_qa/tree/1908d6afbbead072334abe2965f91bd2709910ab)

```bash
python -m pip install -e '.[hotpotqa]'
python -m pip install -r requirements-model.txt
export PYTHONPATH="$MENTAT_SRC${PYTHONPATH:+:$PYTHONPATH}"

jev rag prepare --source validation.parquet \
  --manifest data/hotpotqa/split.json --output data/hotpotqa

for mode in fixed qwen jev; do
  config="configs/hotpotqa_pipeline_${mode}.json"
  jev rag run --config "$config" --tasks data/hotpotqa/tasks.json \
    --manifest data/hotpotqa/split.json --base "$BASE" --checkpoint "$JOINT_CKPT" \
    --output "runs/hotpotqa-$mode"
  jev rag score --predictions "runs/hotpotqa-$mode/predictions.json" \
    --gold data/hotpotqa/gold.json --output "runs/hotpotqa-$mode/metrics.json"
done
```

`trace/` records gate states, option scores, actions, retrieved documents, and draft updates by question. `predictions.json` reports answers, tokens, calls, and time. Scoring reads the answer key separately to compute normalized exact match and answer-token F1. The scoring code comes from HotpotQA under Apache 2.0. [Upstream scoring code](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py)

## HotpotQA 400 reference results

| Policy | EM | F1 | Qwen calls | Jev calls | Input tokens | Output tokens | End-to-end s/question |
|---|---:|---:|---:|---:|---:|---:|---:|
| No gate | 52.75% | 66.46% | 3,706 | 0 | 2,400,254 | 104,250 | 14.62 |
| Qwen gate | 37.25% | 47.74% | 2,348 | 0 | 1,065,085 | 23,849 | 3.17 |
| Jev 95% | 52.25% | 65.52% | 2,602 | 1,818 | 2,758,917 | 64,446 | 10.32 |

The reference combines two existing batches of 200 questions. Timings come from the original runs and depend on caching, scheduling, and hardware. Qwen calls include generation and gating. Jev calls include sufficiency and necessity judgments. Input counts reflect the tokens each model actually reads. The reproduction commands generate per-question predictions and traces. [Summary data](../results/hotpotqa400.json)

The ablation report compares thresholds and compact sufficiency on the same questions. [Gate ablations](HOTPOTQA400_GATE_ABLATION.md) · [Rationale for gating](CONDITIONAL_INTERACTION.md)
