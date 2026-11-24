# Training reproduction

Each joint run initializes from the original Mentat v0.3 adapter and FP32 readout. The Qwen3.5-4B backbone is frozen. Training updates LoRA with rank 64, alpha 128, and dropout 0, plus the native readout. Each state has unit weight. Every configuration runs one epoch with seed 42. Native input tensors are retained. The prompt includes `unknown`, while probabilities are normalized over the given options.

$$
\mathcal L = -\log p_y + 0.5\sum_{k}(p_k-\mathbf 1[k=y])^2.
$$

Training uses gradient accumulation, a cosine schedule with 3% warmup, and gradient clipping at 1.0. It saves resumable checkpoints every 25 updates, verifies that the final checkpoint reloads, then removes optimizer state.

## Joint model grid

v1 and v2 each compare nine combinations of learning rates 5e-6, 1e-5, and 2e-5 and effective batches 16, 32, and 64. Each run shuffles the two task datasets from the same version and trains for one epoch. The selection score weights sufficiency and necessity validation accuracy 60% and 40%. Ties use weighted macro-F1 with the same weights, then smaller batch, lower learning rate, and configuration ID. The selected configuration is frozen before testing.

## v3 supervision from paired rollouts

The v3 design draws on Wang et al.'s (2026) conditional interaction framework, especially selective tool use in Section 6.3 and selective supervision in Section 7.2. Jev screens follow-ups before execution, RAG executes the actions, and offline replay supplies answer quality and call costs for supervision. Forking also executes actions skipped by the original gate so their outcomes can be observed. [Paper](https://arxiv.org/abs/2610.39497) · [Rationale for gating](CONDITIONAL_INTERACTION.md)

Both execution and skipping are tested from the same question, evidence, draft, candidate follow-up, and remaining budget. Both branches use the same downstream rules. Final EM/F1 and call costs are recorded offline. The initial design takes pairs with one correct branch and one incorrect branch and uses the successful action as the binary LoRA target. Pairs where both branches are correct or both are wrong remain in the outcomes and are omitted from these labels. Each labeled pair produces one training sample whose input includes only the pre-fork state. All states and replays for one question share a split.

This design applies the paper's framework to the repository's iterative RAG pipeline. v3 release is pending.

## Single-task training interface

Use this command sequence to train a single task on custom data. Set `BASE` and `MENTAT_SRC`, and point `INITIAL` to the original Mentat v0.3 checkpoint before fine-tuning either task. Training uses one CUDA GPU and needs checkpoint space.

```bash
export PYTHONPATH="$MENTAT_SRC${PYTHONPATH:+:$PYTHONPATH}"
jev prepare-train --config configs/sufficiency.json --output runs/suff \
  --base "$BASE" --initial "$INITIAL" --mentat-src "$MENTAT_SRC"
python runs/suff/code/train.py --tokenize
CUDA_VISIBLE_DEVICES=0 python runs/suff/code/train.py \
  --job '{"id":"b16_lr2e-05","batch":16,"lr":2e-5}'
```

For necessity, use `configs/necessity.json` and a separate run directory. `prepare-train` verifies data, model weights, and source dependencies, then freezes the run manifest. Load the resulting adapter and readout from `training/<id>/latest`.
