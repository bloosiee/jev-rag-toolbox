# 训练复现

每组联合训练从原始 Mentat v0.3 adapter 和 FP32 readout 初始化。冻结 Qwen3.5-4B 主干，更新 rank 64、alpha 128、dropout 0 的 LoRA 与原生 readout。各训练状态权重为 1，每组遍历一轮，seed 42。保留原生输入张量，`unknown` 保留在提示里，概率归一化只使用给定选项。

$$
\mathcal L = -\log p_y + 0.5\sum_{k}(p_k-\mathbf 1[k=y])^2.
$$

优化采用梯度累积、余弦学习率、3% 预热与梯度裁剪 1.0。每 25 次更新保留可恢复检查点，完成后验证重载并清理优化器状态。

## 联合模型网格

v1 和 v2 均比较学习率 5e-6、1e-5、2e-5 与有效 batch 16、32、64 的九种组合。每组将同版本的两个训练任务混洗后训练一轮。按充分性验证准确率的 60% 与必要性验证准确率的 40% 相加选参。并列时依次比较同权重的 macro-F1、更小的 batch、更低的学习率及配置 ID。冻结入选配置后再进行测试。

## v3 分叉结果监督

v3 的训练设计参考 Wang 等（2026）提出的条件交互框架，尤其是第 6.3 节的选择性工具调用与第 7.2 节的选择性监督。Jev 对候选追问作执行前判断，迭代 RAG 执行动作，离线回放得到的答案质量和调用成本再用于监督 Jev。分叉覆盖原门控会跳过的动作，以观察这些动作实际执行后的结果。 [Does Scaling Toward AGI Depend on What System-1 Lets System-2 Execute?](https://arxiv.org/abs/2610.39497) · [理论依据](CONDITIONAL_INTERACTION.md)

在同一原问题、证据、草稿、候选追问和剩余预算上分别执行与跳过该追问。之后两路沿用同一管线规则，离线用标准答案计算最终 EM/F1，并分别记录调用成本。首版使用一边答对、一边答错的配对，将答对分支的动作作为 LoRA 的二分类标签。两路都对或都错的配对保留结果，暂不用于这个胜负标签。每对分叉生成一条训练样本，Jev 的输入只包含分叉前的状态。同题的状态及回放继承同一个训练、验证或测试划分。

该方案是论文框架在本仓库迭代 RAG 上的应用。v3 处于计划阶段。

## 任务级训练接口

以下命令用于自定义单任务训练。

设置 `BASE`、`MENTAT_SRC`。将 `INITIAL` 指向未做这两项微调的原始 Mentat v0.3 模型目录。训练使用单张 CUDA 卡，预留检查点空间。

```bash
export PYTHONPATH="$MENTAT_SRC${PYTHONPATH:+:$PYTHONPATH}"
jev prepare-train --config configs/sufficiency.json --output runs/suff \
  --base "$BASE" --initial "$INITIAL" --mentat-src "$MENTAT_SRC"
python runs/suff/code/train.py --tokenize
CUDA_VISIBLE_DEVICES=0 python runs/suff/code/train.py \
  --job '{"id":"b16_lr2e-05","batch":16,"lr":2e-5}'
```

必要性任务使用 `configs/necessity.json`，输出到新的运行目录。`prepare-train` 校验数据、权重、源码依赖并冻结运行清单。训练结束后从 `training/<id>/latest` 载入相应 adapter/readout。
