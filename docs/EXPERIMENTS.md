# Matched experiments and node replay

## Inspect published experiments

Run these commands from the repository root. They read published results and trace metadata without a GPU or API credentials.

```bash
python -m pip install -e .
jev rag compare --published results/hotpotqa400.json
jev rag compare --published results/2wikimultihopqa400.json
jev rag diagnose --input results/hotpotqa400_gate_traces.jsonl.gz \
  --id 5ab9535a5542996be202047e --arms 80_80,95_80
```

[Jev v3 results and trace examples](../results/JEV_V3.md)

Historical tables retain the original cost accounting and model settings. [Historical experiment settings](ITERATIVE_RAG.md)

## Offline three-hop demo

Run the demo from the repository root. All three policies share a generator and initial draft. Deterministic adapters simulate the gates.

```bash
python -m pip install -e .
python examples/replay_demo.py --output runs/glass-meridian-demo
```

The command prints the answer, EM, generation calls, and gate calls for each policy. The output directory contains the full comparison in `comparison/comparison.md`, the first action difference in `diagnosis.json`, and the sufficiency and necessity branches in `replay-stop/replay.json` and `replay-skip/replay.json`. Both replay reports include each option's post-fork costs.

```bash
jev rag trace --input runs/glass-meridian-demo/comparison/jev \
  --id demo-glass-meridian --step 12
```

The [demo adapter](../examples/adapters/replay_demo_adapter.py) shows how to plug a generator or gate into the comparison. [Custom inputs and adapters](CUSTOM_INPUTS.md)

## Reproduce the HotpotQA 400 comparison

Run these commands from the repository root in a CUDA environment. The configuration compares no gate, the Qwen gate, and joint Jev v1 at 95/95. It verifies model and dataset versions against the frozen manifest.

### Prepare the data and models

- Set `HOTPOT_DEV` to the pinned HotpotQA distractor validation Parquet. [Pinned dataset revision](https://huggingface.co/datasets/hotpotqa/hotpot_qa/tree/1908d6afbbead072334abe2965f91bd2709910ab/distractor)
- Set `BASE` to the pinned Qwen3.5-4B directory and `MENTAT_SRC` to the Mentat source tree's `src` directory. [Model card and base revision](../models/README.md#download-v1-weights)
- Download and extract the joint v1 weights. `CKPT` points to the directory containing the adapter and readout.

```bash
python -m pip install -e '.[hotpotqa]'
python -m pip install -r requirements-model.txt
export PYTHONPATH="$MENTAT_SRC${PYTHONPATH:+:$PYTHONPATH}"

python scripts/download_weights.py jev-joint-v1
tar -xzf checkpoints/joint.tar.gz -C checkpoints
CKPT="$PWD/checkpoints/joint"

INPUT=runs/hotpotqa400-input
jev rag prepare --source "$HOTPOT_DEV" \
  --manifest data/hotpotqa/split.json --output "$INPUT"
```

### Run the three policies

```bash
RUN=runs/hotpotqa400-comparison
jev rag compare --config configs/hotpotqa_compare.json \
  --tasks "$INPUT/tasks.json" --gold "$INPUT/gold.json" \
  --manifest data/hotpotqa/split.json \
  --base "$BASE" --checkpoint "$CKPT" --output "$RUN"
```

Inspect `$RUN/comparison.md` and `$RUN/comparison.json`. The three policies share an initial draft and each is charged its generation cost. The table reports EM, F1, tokens, calls, and time. [Historical methodology and results](ITERATIVE_RAG.md)

### Compare a divergence and replay it

Pick a question appearing in both `$RUN/fixed/predictions.json` and `$RUN/jev/predictions.json`. Set `QUESTION_ID` to its ID. Use the trace to find the decision you want to change and set `NODE` to its `step` value.

```bash
jev rag diagnose --input "$RUN/fixed" "$RUN/jev" \
  --id "$QUESTION_ID" --arms fixed,jev
jev rag trace --input "$RUN/jev" --id "$QUESTION_ID"
jev rag replay --run "$RUN/jev" --tasks "$INPUT/tasks.json" \
  --question-id "$QUESTION_ID" --node "$NODE" --options all \
  --gold "$INPUT/gold.json" --output "$RUN/replay-$QUESTION_ID"
```

Open `replay.json` to compare each option's final answer, EM/F1, and additional cost after the fork.

## Reproduce the HotpotQA gate ablations

Reuse `INPUT`, `BASE`, and `CKPT` from above. The configurations are full-input 95/80, full-input 80/80, and compact sufficiency at 80/80.

```bash
for mode in 95_80 80_80 80_80_compact; do
  out="runs/hotpotqa-$mode"
  jev rag run --config "configs/hotpotqa_jev_$mode.json" \
    --tasks "$INPUT/tasks.json" --manifest data/hotpotqa/split.json \
    --base "$BASE" --checkpoint "$CKPT" --output "$out"
  jev rag score --predictions "$out/predictions.json" \
    --gold "$INPUT/gold.json" --output "$out/metrics.json"
done
```

## Run a 2Wiki or custom comparison

The 2Wiki split manifest verifies the pinned dev Parquet. Reuse the model settings above and select the 2Wiki configuration. [Pinned 2Wiki dev Parquet](https://huggingface.co/datasets/ohjoonhee/2WikiMultihopQA/blob/7395fc70e1b8e70b2cee4376e39500f9c20555d2/data/validation-00000-of-00001.parquet)

```bash
jev rag prepare --manifest data/2wikimultihopqa/split.json \
  --source /path/to/2wiki-dev.parquet --output runs/2wiki-input
jev rag compare --config configs/2wikimultihopqa_compare.json \
  --tasks runs/2wiki-input/tasks.json --gold runs/2wiki-input/gold.json \
  --base "$BASE" --checkpoint "$CKPT" --output runs/2wiki-comparison
```

The 2Wiki configuration also includes `rule`. It skips a follow-up when its normalized terms and term counts match an executed follow-up, without a model call for the gate. Literal duplicate suppression in the controller applies to all policies.

Use the [converters and adapters](CUSTOM_INPUTS.md) to replace inputs or model backends. The comparison accepts `--generator-adapter`, `--gate-adapter`, and `--qwen-gate-adapter` for the shared generator, Jev gate, and Qwen gate. It reads the answer key after all predictions are saved.

## Inspect step-by-step traces

`trace` accepts public trace indexes and local run directories. Omit the question ID to list questions and policies. Step 0 is the shared draft, and step 1 is usually the first sufficiency decision.

```bash
TRACE=results/hotpotqa400_gate_traces.jsonl.gz
jev rag trace --input "$TRACE"
jev rag trace --input "$TRACE" --id "$QUESTION_ID" --arm compact_80_80 --step 1
```

The public index includes gate probabilities, actions, retrieved document indices, call costs, and hashes. Local runs also preserve full states, prompts, outputs, and drafts. [Results and public trace indexes](../results/README.md)

## Replay requirements

`--options` accepts `all` or a comma-separated list of the selected node's options. Replay verifies and reuses the recorded prefix, forces one action, and reruns the suffix with the original controller and backend. Reuse the original adapter configuration and keep endpoint and weight versions fixed. The configuration hash verifies the file contents. The caller pins the version served by an endpoint.

The answer key enables EM/F1 reporting. Including the original option reruns it as a cost reference. Post-fork timing excludes model loading and the shared prefix. A single replay with a stochastic backend records one outcome.

Live replay requires a full Jev trace recorded by the current `jev rag run` or `jev rag compare`. The controller supports custom model backends. A separate RAG controller needs an interface for restoring state and overriding actions.
