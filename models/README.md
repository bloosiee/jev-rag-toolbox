# 单任务模型卡

两个模型从原始 Mentat v0.3 的 LoRA 适配器和 FP32 读出层分别初始化并独立训练，基座为 Qwen3.5-4B。

| 模型 | 判断目标 | 入选配置 | 测试准确率 | 原始 Mentat v0.3 | 补充指标 |
| --- | --- | --- | --- | --- | --- |
| `sufficiency` | 检索证据是否足够回答原问题 | batch 16，学习率 2e-5 | 2,014 / 2,138（94.20%） | 1,689 / 2,138（79.00%） | AUROC 0.9809 |
| `necessity` | 执行追问、证据已覆盖或对原问题无帮助 | batch 64，学习率 1e-5 | 337 / 396（85.10%） | 310 / 396（78.28%） | macro-F1 0.7962 |

分数按各自测试集的决策状态计算。充分性使用 SufficiencyBench，必要性分数衡量与教师标签的一致性。 [数据卡](../docs/DATASETS.md)

## 下载权重

从 [单任务模型 Release](https://github.com/bloosiee/jev-rag-toolbox/releases/tag/single-task-models-v1) 下载，脚本校验各分卷和组装后的压缩包。

```bash
python scripts/download_weights.py sufficiency
python scripts/download_weights.py necessity
tar -xzf checkpoints/sufficiency.tar.gz -C checkpoints
tar -xzf checkpoints/necessity.tar.gz -C checkpoints
```

将 `checkpoints/sufficiency` 或 `checkpoints/necessity` 传给相应任务的 `--checkpoint`。压缩包内含 LoRA、决策读出层、配置和许可。 [充分性 SHA-256](sufficiency.sha256.json) · [必要性 SHA-256](necessity.sha256.json) · [Qwen3.5-4B 基座](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a)
