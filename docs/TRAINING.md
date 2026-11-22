# 训练复现

两个任务各自从原始 Mentat v0.3 adapter 和 FP32 readout 初始化。冻结 Qwen3.5-4B 主干，更新 rank 64、alpha 128、dropout 0 的 LoRA 与原生 readout。各训练状态权重为 1，每组遍历一轮，seed 42。保留原生输入张量，`unknown` 保留在提示里，概率归一化只使用给定选项。

$$
\mathcal L = -\log p_y + 0.5\sum_{k}(p_k-\mathbf 1[k=y])^2.
$$

优化采用梯度累积、余弦学习率、3% 预热与梯度裁剪 1.0。每 25 次更新保留可恢复检查点，完成后验证重载并清理优化器状态。

## 准备和运行

设置 `BASE`、`MENTAT_SRC`。将 `INITIAL` 指向未做这两项微调的原始 Mentat v0.3 模型目录。训练使用单张 CUDA 卡，预留检查点空间。

```bash
export PYTHONPATH="$MENTAT_SRC${PYTHONPATH:+:$PYTHONPATH}"
jev prepare-train --config configs/sufficiency.json --output runs/suff \
  --base "$BASE" --initial "$INITIAL" --mentat-src "$MENTAT_SRC"
python runs/suff/code/train.py --tokenize
CUDA_VISIBLE_DEVICES=0 python runs/suff/code/train.py \
  --job '{"id":"b16_lr2e-05","batch":16,"lr":2e-5}'
```

必要性使用 `configs/necessity.json`，输出到新的运行目录。已选配置为 `{"id":"b64_lr1e-05","batch":64,"lr":1e-5}`。`prepare-train` 校验数据、权重、源码依赖并冻结运行清单。训练结束后从 `training/<id>/latest` 载入相应 adapter/readout。

## 网格与选参

每个任务全部九组独立训练。学习率为 5e-6、1e-5、2e-5，有效 batch 为 16、32、64。对各配置用 `jev eval --split val` 计分。充分性按 AUROC 排序，必要性按 macro-F1 排序。并列时依次取准确率更高、有效 batch 更小、学习率更低、配置 ID 字典序更靠前的配置。记录入选配置后只用该配置跑测试集。
