# 对比实验与逐节点回放

## 查看已有实验

在仓库根目录执行。以下命令读取已发布结果或轨迹元数据，可在无 GPU、无 API 凭据的环境运行。

```bash
python -m pip install -e .
jev rag compare --published results/hotpotqa400.json
jev rag compare --published results/2wikimultihopqa400.json
jev rag diagnose --input results/hotpotqa400_gate_traces.jsonl.gz \
  --id 5ab9535a5542996be202047e --arms 80_80,95_80
```

历史表保留原实验的成本口径和模型配置。轨迹索引含门控概率、执行动作和文档序号，完整问题状态、草稿、答案和实时分叉需要自行运行保存。 [历史实验设置](ITERATIVE_RAG.md)

## 一次运行多个策略

先运行 `jev rag prepare` 生成独立的 `tasks.json` 和 `gold.json`。HotpotQA 的固定输入可用 [HotpotQA 示例](ITERATIVE_RAG.md)准备。2Wiki 使用 `data/2wikimultihopqa/split.json` 和对应的 [固定版本 dev Parquet](https://huggingface.co/datasets/ohjoonhee/2WikiMultihopQA/blob/7395fc70e1b8e70b2cee4376e39500f9c20555d2/data/validation-00000-of-00001.parquet)。安装 `pyarrow` 后可直接读取 Parquet。自定义数据可通过[转换器](CUSTOM_INPUTS.md)给出相同格式。

```bash
jev rag prepare --manifest data/2wikimultihopqa/split.json \
  --source /path/to/2wiki-dev.parquet --output runs/2wiki-input
jev rag compare --config configs/2wikimultihopqa_compare.json \
  --tasks runs/2wiki-input/tasks.json --gold runs/2wiki-input/gold.json \
  --base "$BASE" --checkpoint "$CKPT" --output runs/2wiki-comparison
```

HotpotQA 使用 `configs/hotpotqa_compare.json`，包含无门控、Qwen 门控和 Jev 三组。2Wiki 配置另加 `rule`，当候选追问与已执行追问在去停用词后具有相同词项及频次时跳过，该规则没有模型调用。控制器对字面重复追问的去重适用于所有组。新运行由配置定义，历史运行的阈值和计时不通过新配置反推。

各组共用首轮检索及草稿，从后续决策开始各自运行。所有预测落盘后才读取答案键。输出含 `comparison.json`、`comparison.md`、各组预测和各步轨迹。首轮生成成本计入每组，模型加载不计入；按原生模型和外部适配器分别记录调用来源。用 `--generator-adapter` 替换共享生成后端，用 `--gate-adapter` 替换 Jev 门控。Qwen 对照默认走原生实现，也可以用 `--qwen-gate-adapter` 指定自己的门控实现。适配器从 `--adapter-config` 读取配置。

## 对同题做诊断或回放

```bash
jev rag diagnose --input runs/2wiki-comparison/fixed runs/2wiki-comparison/jev \
  --id "$QUESTION_ID" --arms fixed,jev

jev rag trace --input runs/2wiki-comparison/jev --id "$QUESTION_ID"
jev rag replay --run runs/2wiki-comparison/jev --tasks runs/2wiki-input/tasks.json \
  --question-id "$QUESTION_ID" --node 003_necessity --options all \
  --gold runs/2wiki-input/gold.json --output runs/replay-example
```

用 `trace` 里的实际决策编号替换示例编号。`--options` 接受 `all` 或逗号分隔的该节点选项。回放逐步核对并复用保存的前缀，只强制指定节点的动作，再由原控制器和后端重跑后续轨迹。外部适配器需提供原配置文件，模型端点和权重应保持一致。保存的配置摘要能验证配置文件内容；服务端模型版本由运行者固定。带答案键时报告答案 EM/F1；包含原选项时，额外对照重跑原选项的成本。后缀计时排除模型加载和共享前缀。随机后端的单次回放反映这次观测。

实时回放使用当前版本 `jev rag run` 或 `jev rag compare` 保存的 Jev 轨迹。它支持替换现有控制器的模型后端；接入独立 RAG 控制器需要该控制器提供状态恢复和动作覆盖。
