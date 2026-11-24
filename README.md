# Jev-RAG Toolbox single-task models

This branch contains independently trained sufficiency and necessity models. Each bundle includes a LoRA adapter and decision readout. Joint gates and iterative RAG experiments are on the [main branch](https://github.com/bloosiee/jev-rag-toolbox/tree/main).

| Decision | Input | Output |
| --- | --- | --- |
| Draft sufficiency | Original question, retrieved documents, draft, follow-up history, and pending questions | `sufficient` / `continue` |
| Follow-up necessity | Original question, retrieved documents, executed follow-ups, draft, pending questions, and candidate follow-up | `execute` / `skip_covered` / `skip_unhelpful` |

In iterative retrieval, sufficiency can trigger final answer generation and `execute` can trigger retrieval. Retrieved evidence is added to the next state. Jev returns a choice and probabilities, while the controller handles retrieval, generation, and stopping.

## Quick start

```bash
python -m pip install --upgrade 'pip>=23'
python -m pip install -e .
jev predict --task necessity --method rule --request examples/necessity.json
```

Jev and Qwen use the native Mentat prompt format. Set `BASE` to the Qwen3.5-4B directory, `CKPT` to the adapter and readout directory, and `MENTAT_SRC` to the Mentat source tree's `src` directory.

```bash
python -m pip install -r requirements-model.txt
export PYTHONPATH="$MENTAT_SRC${PYTHONPATH:+:$PYTHONPATH}"
jev predict --task trajectory_sufficiency --request examples/trajectory_sufficiency.json \
  --base "$BASE" --checkpoint "$CKPT"
```

The available methods are `jev`, `qwen`, and `rule`. Qwen disables the adapter and selects the highest-scoring option letter for the same prompt. The rule always continues retrieval or executes the follow-up. Evaluation writes `predictions.json` before reading labels and writing `metrics.json`. Metrics include accuracy, macro-F1, the confusion matrix, and sufficiency AUROC when probabilities are available.

## Weights, data, and training

- [Model cards and downloads](models/README.md)
- [Data card](docs/DATASETS.md)
- [Single-task training interface](docs/TRAINING.md#single-task-training-interface)
- [Custom inputs and adapters](docs/CUSTOM_INPUTS.md)
