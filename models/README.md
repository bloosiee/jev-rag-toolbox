# 联合模型卡

| 版本 | 充分性训练 | 必要性训练 | 状态 |
| --- | --- | --- | --- |
| v1 | SufficiencyBench v1.1，2,704 个状态 | RAG-Necessity-987，1,822 个教师标注状态 | 已发布，batch 16，学习率 2e-5 |
| v2 | 真实轨迹的草稿充分性，927 个状态 | 同 v1 | 训练中 |
| v3 | 以 v2 为对照 | 按 Wang 等的条件交互框架，用执行和跳过的终局结果生成动作标签 | [训练方案](../docs/TRAINING.md#v3-分叉结果监督) · 计划中 |

v1 对应现有 [400 题端到端评估](../docs/ITERATIVE_RAG.md)及[门控消融](../docs/HOTPOTQA400_GATE_ABLATION.md)。v2 在固定划分上按充分性验证准确率 60%、必要性验证准确率 40% 选择联合模型。

## 下载 v1 权重

[GitHub Release](https://github.com/bloosiee/jev-rag-toolbox/releases/tag/v0.1.0-models) 提供 v1 联合模型的分卷。脚本依次核验分卷和压缩包的 SHA-256。

```bash
python scripts/download_weights.py v1
tar -xzf checkpoints/joint.tar.gz -C checkpoints
```

将 `checkpoints/joint` 传给 `--checkpoint`。权重目录包含 LoRA、决策读出层及配置，详见[文件校验清单](joint.sha256.json)。基座从上游获取。 [Qwen3.5-4B 固定版本](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a) · [Mentat v0.3 初始化模型](https://huggingface.co/yunqu/mentat-sys1-v0.3)
