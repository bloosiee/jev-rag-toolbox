# Jev-RAG Toolbox

训练和评测面向具体决策的 Jev。v0.1 提供两个独立训练的决策头，配套同一套推理和评测命令。

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

## DRIFT-inspired BM25 RAG

借鉴微软 GraphRAG 的 DRIFT Search 追问流程，在每题自带的候选文档上用 BM25 检索。三组共享草稿和追问逻辑，分别使用无门控、Qwen 门控和 Jev 门控。每次运行生成逐题答案、决策轨迹和成本记录。 [HotpotQA 400 题复现与结果](docs/ITERATIVE_RAG.md) · [门控消融](docs/HOTPOTQA400_GATE_ABLATION.md) · [查询逐步轨迹](docs/GATE_TRACES.md)

## 自定义输入

用 `jev convert` 导入决策日志或 RAG 测试集。用 Python 适配器接入自己的生成模型或门控模型。 [自定义输入与适配器示例](docs/CUSTOM_INPUTS.md)

## 训练与结果

- [数据卡](docs/DATASETS.md)记录两套训练数据的来源、字段、标签及冻结划分。
- [训练复现](docs/TRAINING.md)提供训练命令、目标函数、网格和验证选参规则。
- [模型卡](models/README.md)记录已有模型配置、评测成绩和权重校验方式。
- [仓库约定](docs/CONVENTIONS.md)记录目录设计、实验记录规范和参考仓库。

复现需要原始 Qwen3.5-4B、Mentat v0.3 权重与对应源码。将训练和评测所需的 JSON 放到对应的 `data/<task>/` 目录。用 `data/*/FILES_SHA256.json` 核对版本。模型卡列出了权重校验值。
