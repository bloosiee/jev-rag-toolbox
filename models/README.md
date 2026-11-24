# Single-task model cards

| Model | Decision | Training and settings |
| --- | --- | --- |
| `sufficiency` | Whether the draft fully answers the original question with support from retrieved evidence | 927 recorded iterative RAG states, batch 16, learning rate 2e-5 |
| `necessity` | Whether to execute a follow-up, skip it because evidence already covers it, or skip it because it does not help answer the question | 1,822 necessity states, batch 64, learning rate 1e-5 |

Both models initialize separately from Mentat v0.3 on Qwen3.5-4B.

## Draft sufficiency test

| Model | Accuracy | AUROC | Incorrect drafts accepted |
| --- | ---: | ---: | ---: |
| Original Mentat v0.3 | 182 / 196 (92.86%) | 0.9847 | 12 / 49 (24.49%) |
| Earlier standalone sufficiency | 177 / 196 (90.31%) | 0.9679 | 8 / 49 (16.33%) |
| Trajectory-trained standalone sufficiency | 195 / 196 (99.49%) | 0.9999 | 0 / 49 (0.00%) |

All three models use the same trajectory requests and input tensors. The test covers a subset of trajectories sourced from the HotpotQA training split that can be reliably labeled by the rules. Metrics are per state. The necessity model scores 337 / 396 (85.10%) with macro-F1 of 0.7962 on its original test set.

## Download weights

The script verifies the individual parts and the assembled archive. Sufficiency and necessity have separate releases. [Trajectory sufficiency](https://github.com/bloosiee/jev-rag-toolbox/releases/tag/single-task-models-v2) · [Necessity](https://github.com/bloosiee/jev-rag-toolbox/releases/tag/single-task-models-v1)

```bash
python scripts/download_weights.py sufficiency
python scripts/download_weights.py necessity
tar -xzf checkpoints/sufficiency.tar.gz -C checkpoints
tar -xzf checkpoints/necessity.tar.gz -C checkpoints
```

For sufficiency, use `jev predict --task trajectory_sufficiency` with `checkpoints/sufficiency`. The request includes the draft and follow-up history. For necessity, use `--task necessity` with `checkpoints/necessity`. [Example trajectory request](../examples/trajectory_sufficiency.json) · [Sufficiency checksums](sufficiency.sha256.json) · [Necessity checksums](necessity.sha256.json)
