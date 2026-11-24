# Repository conventions

## Directory layout

| Directory | Contents |
| --- | --- |
| `docs/` | Guides, data cards, and experiment methodology |
| `configs/` | Gate, training, and evaluation configurations |
| `data/` | Dataset splits and verification manifests |
| `models/` | Model cards and checkpoint download manifests |
| `results/` | Published results, figures, and trace indexes |
| `examples/` | Runnable examples, adapters, and inputs |
| `jev_toolbox/` | CLI and core implementation |
| `recipes/`, `scripts/`, `tests/` | Training recipes, release scripts, and checks |
| `runs/` | Local artifacts stored in separate run directories |

## Experiment records

- Use one CLI and keep data paths, training recipes, and selection criteria in separate configurations. The layout follows the entry point and configurations in [LAMDA-PILOT](https://github.com/LAMDA-CL/LAMDA-PILOT).
- Use the same task configuration when switching evaluation methods. Freeze and store requests and labels separately, and write predictions and metrics for each run. The configuration approach draws on [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
- Record data sources, labels, splits, and limitations in data cards. Record versions, input protocols, training methods, metrics, and checksums for models. [Hugging Face dataset cards](https://huggingface.co/docs/hub/datasets-cards) · [Model cards](https://huggingface.co/docs/hub/model-cards)
- Freeze SHA-256 hashes for inputs and checkpoints and keep each run in a new directory. Log decisions, actions, and state updates separately so each can be inspected.
