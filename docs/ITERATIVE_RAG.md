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

[复现 HotpotQA 400 题三组对照](EXPERIMENTS.md#复现-hotpotqa-400-题三组对照) 汇总了数据、权重、对照和回放命令。

`trace/` 按题记录门控状态、选项分数、路由动作、检索文档与草稿更新。`predictions.json` 汇总答案、token、调用数和时间。评分单独读取答案文件，计算标准化精确匹配和答案词元 F1。评分实现取自 HotpotQA 的答案评测代码，适用附带的 Apache 2.0 许可。 [上游评测代码](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py)

## 实验结果

[主实验、消融与轨迹](../results/README.md)
