# DRIFT-inspired BM25 RAG

方法参考微软 GraphRAG 的 DRIFT Search。DRIFT 使用社区报告生成初始答案和追问，再沿知识图谱开展局部搜索。本实验借鉴多轮追问流程，在每道题自带的候选文档内做 BM25 检索，按候选队列顺序处理追问，并比较三种门控策略。 [Microsoft Research 介绍](https://www.microsoft.com/en-us/research/blog/introducing-drift-search-combining-global-and-local-search-methods-to-improve-quality-and-efficiency/) · [GraphRAG 官方文档](https://microsoft.github.io/graphrag/query/drift_search/)

每道题先用 BM25 取前两篇文档，由 Qwen 写出初始草稿。每轮先判断草稿是否完整且有已见证据支持。若继续检索，Qwen 提出一到三个候选追问。必要性判断逐条决定是否执行，执行时检索三篇文档并用 Qwen 读出分支答案。已执行分支的文档和答案进入状态，随后 Qwen 重写草稿。停搜时返回当前草稿，最多运行三轮。

| 配置 | 充分性 | 必要性 |
|---|---|---|
| `hotpotqa_pipeline_fixed.json` | 按轮次推进 | 执行全部入队追问 |
| `hotpotqa_pipeline_qwen.json` | Qwen 选择候选字母 | Qwen 三分类 |
| `hotpotqa_pipeline_jev.json` | Jev 读出，95% 门槛 | Jev 二分类，95% 门槛 |

Jev 使用经两个任务联合训练的 `batch=16, lr=2e-5` adapter/readout。两项温度由原验证集 NLL 拟合。低置信度时继续检索或执行追问。该门槛结果属于探索性比较。Qwen 与 Jev 的必要性选项数不同，表格比较的是三套完整策略。

## 复现命令

安装模型依赖，设置 `MENTAT_SRC`、`BASE` 和 `JOINT_CKPT`。联合权重和基座的文件校验值列在 `data/hotpotqa/split.json`。下载指定版本的 HotpotQA distractor validation Parquet，或提供官方 distractor dev JSON。准备命令逐题校验公开请求和答案索引。 [HotpotQA 数据与论文](https://hotpotqa.github.io/) · [冻结数据版本](https://huggingface.co/datasets/hotpotqa/hotpot_qa/tree/1908d6afbbead072334abe2965f91bd2709910ab)

```bash
python -m pip install -e '.[hotpotqa]'
python -m pip install -r requirements-model.txt
export PYTHONPATH="$MENTAT_SRC${PYTHONPATH:+:$PYTHONPATH}"

jev rag prepare --source validation.parquet \
  --manifest data/hotpotqa/split.json --output data/hotpotqa

for mode in fixed qwen jev; do
  config="configs/hotpotqa_pipeline_${mode}.json"
  jev rag run --config "$config" --tasks data/hotpotqa/tasks.json \
    --manifest data/hotpotqa/split.json --base "$BASE" --checkpoint "$JOINT_CKPT" \
    --output "runs/hotpotqa-$mode"
  jev rag score --predictions "runs/hotpotqa-$mode/predictions.json" \
    --gold data/hotpotqa/gold.json --output "runs/hotpotqa-$mode/metrics.json"
done
```

`trace/` 按题记录门控状态、选项分数、路由动作、检索文档与草稿更新。`predictions.json` 汇总答案、token、调用数和时间。评分单独读取答案文件，计算标准化精确匹配和答案词元 F1。评分实现取自 HotpotQA 的答案评测代码，适用附带的 Apache 2.0 许可。 [上游评测代码](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py)

## HotpotQA 400 题参考结果

| 策略 | EM | F1 | Qwen 调用 | Jev 调用 | 输入 token | 输出 token | 端到端秒/题 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 无门控 | 52.75% | 66.46% | 3,706 | 0 | 2,400,254 | 104,250 | 14.62 |
| Qwen 门控 | 37.25% | 47.74% | 2,348 | 0 | 1,065,085 | 23,849 | 3.17 |
| Jev 95% | 52.25% | 65.52% | 2,602 | 1,818 | 2,758,917 | 64,446 | 10.32 |

参考结果合并两批各 200 题的既有预测。耗时来自原实验，受缓存、调度和硬件影响。Qwen 调用包含生成与判断，Jev 调用只包含充分性和必要性判断。两阶段的输入均按实际被各模型读取的 token 计数。逐题预测和轨迹由复现命令生成。 [汇总数据](../results/hotpotqa400.json)

附录提供同题的不同门槛及精简充分性对照。 [400 题门控消融](HOTPOTQA400_GATE_ABLATION.md) · [理论依据](CONDITIONAL_INTERACTION.md)
