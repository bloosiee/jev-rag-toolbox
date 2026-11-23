# Jev-RAG Toolbox 单任务模型

此分支收录分别训练的充分性和必要性模型。模型包包含 LoRA 适配器与决策读出层。联合门控模型和迭代 RAG 实验见 [主分支](https://github.com/bloosiee/jev-rag-toolbox/tree/main)。

| 决策 | 输入 | 输出 |
| --- | --- | --- |
| 证据充分性 | 原问题、已检索文档 | `sufficient` / `insufficient` |
| 追问必要性 | 原问题、已检索文档、已执行追问、中间答案、待处理问题、候选追问 | `execute` / `skip_covered` / `skip_unhelpful` |

在迭代搜索中，`sufficient` 可以触发生成最终答案，`execute` 可以触发一次检索。执行后的证据写回状态，供下一轮判断。Jev 返回选项和概率，检索、生成和停止条件由流程代码控制。

## 快速试跑

```bash
python -m pip install --upgrade 'pip>=23'
python -m pip install -e .
jev predict --task necessity --method rule --request examples/necessity.json
```

Jev 和 Qwen 对照共用 Mentat 的原生提示格式。设置 `BASE` 为 Qwen3.5-4B 目录，`CKPT` 为对应 adapter/readout 目录，`MENTAT_SRC` 为 Mentat 源码目录。

```bash
python -m pip install -r requirements-model.txt
export PYTHONPATH="$MENTAT_SRC${PYTHONPATH:+:$PYTHONPATH}"
jev predict --task sufficiency --request examples/sufficiency.json \
  --base "$BASE" --checkpoint "$CKPT"
jev eval --config configs/sufficiency.json --split val --method jev \
  --base "$BASE" --checkpoint "$CKPT" --output runs/suff-jev-val
```

`--method` 支持 `jev`、`qwen`、`rule`。Qwen 禁用 adapter，从相同输入的候选字母 logits 取最大值。固定规则始终继续检索或执行追问。评测先写 `predictions.json`，再读取标签并写 `metrics.json`。指标包括准确率、macro-F1、混淆矩阵，以及有概率输出时的充分性 AUROC。

## 权重、数据与训练

- [模型卡和权重下载](models/README.md)
- [数据卡](docs/DATASETS.md)
- [单任务训练接口](docs/TRAINING.md#任务级训练接口)
- [自定义输入与适配器](docs/CUSTOM_INPUTS.md)
