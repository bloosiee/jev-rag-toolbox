# Custom logs, datasets, and models

## Try and evaluate a single gate

Install the base package from the repository root. The rule baseline runs without a GPU or weights.

```bash
python -m pip install -e .
jev predict --task necessity --method rule --request examples/necessity.json
```

For native models, set `BASE` to Qwen3.5-4B, `CKPT` to the adapter and readout directory, and `MENTAT_SRC` to the Mentat source tree's `src` directory. [Checkpoint and model versions](../models/README.md)

```bash
python -m pip install -r requirements-model.txt
export PYTHONPATH="$MENTAT_SRC${PYTHONPATH:+:$PYTHONPATH}"
jev predict --task sufficiency --request examples/sufficiency.json \
  --base "$BASE" --checkpoint "$CKPT"
jev eval --config configs/sufficiency.json --split val --method jev \
  --base "$BASE" --checkpoint "$CKPT" --output runs/suff-jev-val
```

Single-gate evaluation supports `jev`, `qwen`, and `rule`. Qwen disables the adapter and selects the highest-scoring option letter for the same prompt. The rule policy always continues retrieval or executes the follow-up. Evaluation saves predictions before reading labels to compute accuracy, macro-F1, the confusion matrix, and sufficiency AUROC when probabilities are available.

## Decision logs

Supported inputs are JSON arrays, JSONL, and CSV. A mapping file maps standard fields to source fields, using dot notation for nested values. Use JSON strings for documents, follow-up history, and pending questions in CSV. Omit `--mapping` when the fields already match.

```bash
jev convert decisions --task sufficiency --input examples/custom_decisions.jsonl \
  --mapping examples/decision_mapping.json --labeled --output runs/custom-decisions

jev eval --task sufficiency --requests runs/custom-decisions/requests.json \
  --gold runs/custom-decisions/gold.json --method rule --output runs/custom-decision-eval
```

Sufficiency takes `original_question` and `retrieved_documents`. Necessity also takes `executed_followups` with a `question` and `answer` for each entry, `current_intermediate_answer`, `pending_questions`, and `proposed_followup`. The default label column is `label` and uses the task's fixed option values. Omit `--labeled` and `--gold` for unlabeled prediction.

## RAG datasets

Provide `id`, `question`, and `documents` in each row. Each document needs a `title` and either `text` or `sentences`. For a labeled dataset, include `answer` and pass `--labeled`.

```bash
jev convert rag --input examples/custom_rag.jsonl --labeled --output runs/custom-rag
```

Pass the generated `tasks.json` to `jev rag run --tasks` without the HotpotQA manifest. Use the separate `gold.json` with `jev rag score`. Custom datasets use the HotpotQA answer EM/F1 scorer. Conversion writes a SHA-256 manifest.

## Model adapters

Add your local Python module to `PYTHONPATH` and identify the factory as `module:factory`. The factory takes a configuration dictionary, empty by default or loaded from `--adapter-config`. Load trusted Python modules and read credentials from environment variables.

| Method | Input | Return value |
|---|---|---|
| `predict(kind, request)` | State, decision question, and options in `request` | `choice`, with optional `probabilities` and token counts |
| `generate(prompt, *, max_new_tokens, schema=None)` | Prompt, generation budget, and optional JSON Schema | `text`, with optional `truncated` and token counts |

Use `--adapter` with `jev predict` or `jev eval` for decision logs. For RAG, use `--generator-adapter` to replace generation and `--gate-adapter` to replace gating. Supply the base and checkpoint whenever a native component remains in use. With the Jev policy, the adapter returns normalized probabilities over the listed options and the configured threshold determines the action. The adapter is responsible for external probability calibration. Supply token counts as `input_tokens` and `output_tokens`. Missing counts are recorded as `null`.

```bash
PYTHONPATH=examples/adapters jev rag run --config configs/hotpotqa_pipeline_jev.json \
  --tasks runs/custom-rag/tasks.json --generator-adapter sample_adapter:build \
  --gate-adapter sample_adapter:build --output runs/custom-rag-run

jev rag score --predictions runs/custom-rag-run/predictions.json \
  --gold runs/custom-rag/gold.json --output runs/custom-rag-run/metrics.json
```

The example adapter uses fixed responses to demonstrate the interface. Replace these methods with your model calls to use a real backend.

## Run RAG with your model and inspect traces

`chat_endpoint.py` calls a local model server or API with a Chat Completions interface. Set `generator_model` in `examples/adapters/chat_endpoint.json` to a model served by the endpoint. Optionally set `gate_model` to select a separate gate. If authentication is required, set `JEV_CHAT_API_KEY` in the environment.

Use the example dataset to verify the connection. Change `--input` to load your own JSON, JSONL, or CSV. Add `--mapping` for custom column names.

```bash
export JEV_CHAT_BASE_URL=http://127.0.0.1:8000/v1

jev convert rag --input examples/custom_rag.jsonl --labeled --output runs/my-rag-data

PYTHONPATH=examples/adapters jev rag run \
  --config configs/hotpotqa_pipeline_qwen.json \
  --tasks runs/my-rag-data/tasks.json \
  --generator-adapter chat_endpoint:build \
  --gate-adapter chat_endpoint:build \
  --adapter-config examples/adapters/chat_endpoint.json \
  --output runs/my-rag-traces

jev rag trace --input runs/my-rag-traces
jev rag trace --input runs/my-rag-traces --id question-1
jev rag trace --input runs/my-rag-traces --id question-1 --step 1
```

This example uses the `qwen` policy's direct option selection. The adapter provides both generation and gating, and the endpoint can return a choice alone. For confidence thresholds, switch to the `jev` policy with a gate adapter that returns option probabilities. An external API receives the current retrieval state and documents. The run directory stores traces locally. [Trace fields and steps](EXPERIMENTS.md#inspect-step-by-step-traces)
