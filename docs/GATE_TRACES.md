# Inspect gate traces

`jev rag trace` accepts a local run directory or a published JSONL trace index.

```bash
TRACE=results/hotpotqa400_gate_traces.jsonl.gz
jev rag trace --input "$TRACE"
jev rag trace --input "$TRACE" --id "$QUESTION_ID"
jev rag trace --input "$TRACE" --id "$QUESTION_ID" --arm compact_80_80 --step 1
```

Omit the ID to list questions and policies. Supply an ID to view a full trace or add `--step` to inspect one call. Step 0 is the shared draft. Step 1 is usually the first sufficiency decision. The branch list records the actions and retrieved document indices for each round.

The public index contains options and probabilities, actions, retrieval indices, per-call tokens and time, and request and rendered-prompt hashes. Local runs also preserve the full states, prompts, outputs, and drafts.

```bash
jev rag trace --input runs/hotpotqa-compact --id "$QUESTION_ID" --step 1
```

## Reproduce gate settings on 400 questions

Use the frozen HotpotQA split and joint weights to prepare `tasks.json`. The three configurations are full-input 95/80, full-input 80/80, and compact-sufficiency 80/80. [Data and checkpoint setup](ITERATIVE_RAG.md)

```bash
for mode in 95_80 80_80 80_80_compact; do
  jev rag run --config "configs/hotpotqa_jev_$mode.json" \
    --tasks data/hotpotqa/tasks.json --manifest data/hotpotqa/split.json \
    --base "$BASE" --checkpoint "$JOINT_CKPT" --output "runs/hotpotqa-$mode"
  jev rag score --predictions "runs/hotpotqa-$mode/predictions.json" \
    --gold data/hotpotqa/gold.json --output "runs/hotpotqa-$mode/metrics.json"
done
```

With custom datasets or models, prepare a local run as shown in the [adapter example](CUSTOM_INPUTS.md#run-rag-with-your-model-and-inspect-traces). The same trace command selects a question and step.
