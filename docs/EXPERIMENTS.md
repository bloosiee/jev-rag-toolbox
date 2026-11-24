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

历史表保留原实验的成本口径和模型配置。 [历史实验设置](ITERATIVE_RAG.md)

## 离线三跳示例

在仓库根目录安装基础包并运行。三种策略使用同一生成适配器和同一初稿。门控也由确定性适配器模拟，输出用于验证对比和回放流程。

```bash
python -m pip install -e .
python examples/replay_demo.py --output runs/glass-meridian-demo
```

运行会列出三组的答案、EM、生成和门控调用次数。输出目录里，`comparison/comparison.md` 是完整对比表，`diagnosis.json` 标出第一处追问动作分歧，`replay-stop/replay.json` 和 `replay-skip/replay.json` 分别记录充分性和必要性分叉。每份回放同时保留原选项和改选项的结果，以及分叉后的新增成本。

```bash
jev rag trace --input runs/glass-meridian-demo/comparison/jev \
  --id demo-glass-meridian --step 12
```

示例生成和门控实现见 [适配器代码](../examples/adapters/replay_demo_adapter.py)。可以把自己的生成器或门控接到同一比较流程。 [自定义输入与适配器](CUSTOM_INPUTS.md)

## 复现 HotpotQA 400 题三组对照

下面把数据准备、同题比较和错题回放连起来。配置为无门控、Qwen 门控和联合 Jev v1 95/95，模型和数据版本按冻结清单校验。运行环境需要 CUDA。在仓库根目录执行。

### 准备数据和模型

- 下载固定版本的 HotpotQA distractor validation Parquet，路径设为 `HOTPOT_DEV`。 [固定数据版本](https://huggingface.co/datasets/hotpotqa/hotpot_qa/tree/1908d6afbbead072334abe2965f91bd2709910ab/distractor)
- 将 `BASE` 设为固定的 Qwen3.5-4B 基座目录，将 `MENTAT_SRC` 设为 Mentat 源码的 `src` 目录。 [模型卡和基座版本](../models/README.md#下载-v1-权重)
- 下载并解包联合 v1 权重后，`CKPT` 指向包含适配器和读出层的目录。

```bash
python -m pip install -e '.[hotpotqa]'
python -m pip install -r requirements-model.txt
export PYTHONPATH="$MENTAT_SRC${PYTHONPATH:+:$PYTHONPATH}"

python scripts/download_weights.py jev-joint-v1
tar -xzf checkpoints/joint.tar.gz -C checkpoints
CKPT="$PWD/checkpoints/joint"

INPUT=runs/hotpotqa400-input
jev rag prepare --source "$HOTPOT_DEV" \
  --manifest data/hotpotqa/split.json --output "$INPUT"
```

### 运行三个策略

```bash
RUN=runs/hotpotqa400-comparison
jev rag compare --config configs/hotpotqa_compare.json \
  --tasks "$INPUT/tasks.json" --gold "$INPUT/gold.json" \
  --manifest data/hotpotqa/split.json \
  --base "$BASE" --checkpoint "$CKPT" --output "$RUN"
```

查看 `$RUN/comparison.md` 和 `$RUN/comparison.json`。三组共享初稿，各自计入初稿生成成本。对照表记录 EM、F1、token、调用次数和耗时。 [历史实验口径和结果](ITERATIVE_RAG.md)

### 查看分歧并回放

从 `$RUN/fixed/predictions.json` 和 `$RUN/jev/predictions.json` 选同一道题，把它的 ID 设为 `QUESTION_ID`。在 `trace` 的输出中找到想改的 Jev 门控事件，把它的 `step` 值设为 `NODE`。

```bash
jev rag diagnose --input "$RUN/fixed" "$RUN/jev" \
  --id "$QUESTION_ID" --arms fixed,jev
jev rag trace --input "$RUN/jev" --id "$QUESTION_ID"
jev rag replay --run "$RUN/jev" --tasks "$INPUT/tasks.json" \
  --question-id "$QUESTION_ID" --node "$NODE" --options all \
  --gold "$INPUT/gold.json" --output "$RUN/replay-$QUESTION_ID"
```

查看回放目录的 `replay.json`，比较每个动作的最终答案、EM/F1 和分叉后的新增成本。

## 复现 HotpotQA 门控消融

沿用上面的 `INPUT`、`BASE`、`CKPT`。三份配置依次对应全输入 95/80、全输入 80/80 和精简充分性 80/80。

```bash
for mode in 95_80 80_80 80_80_compact; do
  out="runs/hotpotqa-$mode"
  jev rag run --config "configs/hotpotqa_jev_$mode.json" \
    --tasks "$INPUT/tasks.json" --manifest data/hotpotqa/split.json \
    --base "$BASE" --checkpoint "$CKPT" --output "$out"
  jev rag score --predictions "$out/predictions.json" \
    --gold "$INPUT/gold.json" --output "$out/metrics.json"
done
```

## 运行 2Wiki 或自定义对照

2Wiki 使用 `data/2wikimultihopqa/split.json` 校验固定版 dev Parquet。沿用上面的模型设置，使用 2Wiki 配置跑对照。 [固定版 2Wiki dev 数据](https://huggingface.co/datasets/ohjoonhee/2WikiMultihopQA/blob/7395fc70e1b8e70b2cee4376e39500f9c20555d2/data/validation-00000-of-00001.parquet)

```bash
jev rag prepare --manifest data/2wikimultihopqa/split.json \
  --source /path/to/2wiki-dev.parquet --output runs/2wiki-input
jev rag compare --config configs/2wikimultihopqa_compare.json \
  --tasks runs/2wiki-input/tasks.json --gold runs/2wiki-input/gold.json \
  --base "$BASE" --checkpoint "$CKPT" --output runs/2wiki-comparison
```

2Wiki 配置另加 `rule`。候选追问与已执行追问在去停用词后具有相同词项及频次时，这个规则会跳过，无需调用模型。控制器的字面重复追问去重适用于所有组。

替换生成器或门控时，使用 [转换器与适配器](CUSTOM_INPUTS.md)。`compare` 支持 `--generator-adapter`、`--gate-adapter` 和 `--qwen-gate-adapter`，分别替换共享生成器、Jev 门控和 Qwen 门控。各组的答案全部落盘后才读取答案键。

## 查看逐步轨迹

`trace` 支持公开轨迹索引或本地运行目录。省略题号会列出可查的问题和策略。第 0 步是共享初稿，第 1 步通常是首次充分性判断。

```bash
TRACE=results/hotpotqa400_gate_traces.jsonl.gz
jev rag trace --input "$TRACE"
jev rag trace --input "$TRACE" --id "$QUESTION_ID" --arm compact_80_80 --step 1
```

公开索引提供门控概率、执行动作、检索文档序号、调用成本和哈希。完整状态、提示、模型输出与草稿保存在本地运行目录。 [结果与公开轨迹索引](../results/README.md)

## 回放条件

`--options` 接受 `all` 或逗号分隔的该节点选项。回放逐步核对并复用保存的前缀，只强制指定节点的动作，再由原控制器和后端重跑后续轨迹。外部适配器需提供原配置文件，模型端点和权重应保持一致。保存的配置摘要能验证配置文件内容；服务端模型版本由运行者固定。带答案键时报告答案 EM/F1；包含原选项时，额外对照重跑原选项的成本。后缀计时排除模型加载和共享前缀。随机后端的单次回放反映这次观测。

实时回放使用当前版本 `jev rag run` 或 `jev rag compare` 保存的 Jev 轨迹。它支持替换现有控制器的模型后端；接入独立 RAG 控制器需要该控制器提供状态恢复和动作覆盖。
