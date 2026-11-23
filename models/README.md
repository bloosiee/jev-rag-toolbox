# 单任务模型卡

| 模型 | 判断目标 | 训练与设置 |
| --- | --- | --- |
| `sufficiency` | 当前草稿是否答全原题且有检索证据支持 | 927 条真实迭代轨迹状态，batch 16，学习率 2e-5 |
| `necessity` | 执行追问、证据已覆盖或对原问题无帮助 | 1,822 条必要性状态，batch 64，学习率 1e-5 |

两个模型分别从原始 Mentat v0.3 初始化，基座为 Qwen3.5-4B。

## 草稿充分性测试

| 模型 | 准确率 | AUROC | 错误草稿误放行 |
| --- | ---: | ---: | ---: |
| 原始 Mentat v0.3 | 182 / 196（92.86%） | 0.9847 | 12 / 49（24.49%） |
| 旧独立充分性 | 177 / 196（90.31%） | 0.9679 | 8 / 49（16.33%） |
| 真实轨迹独立充分性 | 195 / 196（99.49%） | 0.9999 | 0 / 49（0.00%） |

三套模型使用相同轨迹请求与输入张量。该测试集覆盖规则能可靠标注的 HotpotQA 训练来源轨迹子集，指标按状态计分。必要性模型在原测试集上为 337 / 396（85.10%），macro-F1 0.7962。

## 下载权重

脚本校验分卷与组装后的压缩包。充分性和必要性使用各自的发布包。 [轨迹充分性](https://github.com/bloosiee/jev-rag-toolbox/releases/tag/single-task-models-v2) · [必要性](https://github.com/bloosiee/jev-rag-toolbox/releases/tag/single-task-models-v1)

```bash
python scripts/download_weights.py sufficiency
python scripts/download_weights.py necessity
tar -xzf checkpoints/sufficiency.tar.gz -C checkpoints
tar -xzf checkpoints/necessity.tar.gz -C checkpoints
```

充分性用 `jev predict --task trajectory_sufficiency` 加载 `checkpoints/sufficiency`，请求包含草稿和追问历史。必要性用 `--task necessity` 加载 `checkpoints/necessity`。 [轨迹请求示例](../examples/trajectory_sufficiency.json) · [充分性 SHA-256](sufficiency.sha256.json) · [必要性 SHA-256](necessity.sha256.json)
