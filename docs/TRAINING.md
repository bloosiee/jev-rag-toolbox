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

### Labeling actions by downstream outcomes

Both branches start at the same RAG state. Sufficiency compares continued retrieval with submitting the current draft. Necessity compares executing and skipping the follow-up. The downstream policy and budget stay fixed. The recorded outcomes include final EM/F1, Qwen calls, input and output tokens, and Qwen and Jev time.

The earlier gates already reduced Qwen calls, so v3 prioritizes final answer quality and uses cost to break ties between successful actions. The main pipeline experiment keeps the v1 and v2 sufficiency and necessity thresholds at 95/95.

| Observed branch outcomes | v3 training target |
| --- | --- |
| One branch is correct | Select the correct action |
| Both are correct with different Qwen call counts | Select the action with fewer calls |
| Both are correct with the same Qwen call count | Keep the outcome pair without a hard label |
| Both are incorrect | Keep the outcome pair without a hard label |

The audit records EM/F1 and costs for each pair. The model sees the pre-fork state in the decision prompt. The selected action serves as the training target. Post-fork answers, scores, costs, and reference answers are used for offline verification. All states from a question share a split. Labels reflect outcomes under the recorded generator, policy, and budget. Pairs with two wrong answers are counted separately.

HotpotQA's official training split supplies 1,000 valid sufficiency pairs and 1,000 valid necessity pairs. The resulting 1,225 labels are stratified by gate and action with seed 42 into 985 train, 120 validation, and 120 test examples. Joint training uses the same nine training configurations. The fork validation score weights sufficiency and necessity 60% and 40%. The selected configuration uses effective batch 64 and learning rate 2e-5. The end-to-end benchmarks reuse the original question sets, generator-specific initial drafts, and search budgets.

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
