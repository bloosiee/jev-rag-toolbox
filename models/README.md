# Joint model cards

Toolbox releases use `toolbox-vX.Y.Z`. Joint checkpoints and future checkpoint releases use `jev-joint-vN`. Earlier download URLs remain available.

| Checkpoint | Sufficiency training | Necessity training | Status |
| --- | --- | --- | --- |
| `jev-joint-v1` | SufficiencyBench v1.1, 2,704 states | RAG-Necessity-987, 1,822 teacher-labeled states | [Download weights](#download-v1-weights), batch 16, learning rate 2e-5 |
| `jev-joint-v2` | Draft sufficiency from recorded trajectories, 927 states | Same as v1 | [Download weights](#download-v2-weights), batch 16, learning rate 2e-5 |
| `jev-joint-v3` | Outcome labels from stopping or continuing retrieval | Outcome labels from executing or skipping a follow-up | [Training method](../docs/TRAINING.md#v3-supervision-from-paired-rollouts), batch 64, learning rate 2e-5 · Weights pending |

`jev-joint-v2` was selected on a fixed split using validation accuracy weighted 60% for sufficiency and 40% for necessity. [Results and traces](../results/README.md)

## Download v1 weights

The weight parts are available in the [jev-joint-v1 Release](https://github.com/bloosiee/jev-rag-toolbox/releases/tag/jev-joint-v1). The script verifies the SHA-256 of each part and of the assembled archive.

```bash
python scripts/download_weights.py jev-joint-v1
tar -xzf checkpoints/joint.tar.gz -C checkpoints
```

Pass `checkpoints/joint` to `--checkpoint`. The directory contains LoRA weights, a decision readout, and configuration. Obtain the base model from upstream. [Checkpoint checksums](joint.sha256.json) · [Pinned Qwen3.5-4B revision](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a) · [Mentat v0.3 initialization](https://huggingface.co/yunqu/mentat-sys1-v0.3)

## Download v2 weights

The weight parts for `jev-joint-v2` are available in the [joint-model-v2 Release](https://github.com/bloosiee/jev-rag-toolbox/releases/tag/joint-model-v2).

```bash
python scripts/download_weights.py jev-joint-v2
tar -xzf checkpoints/joint-v2.tar.gz -C checkpoints
```

Pass `checkpoints/joint-v2` to `--checkpoint`. [Checkpoint checksums](joint-v2.sha256.json)
