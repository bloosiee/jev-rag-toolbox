# 查看门控轨迹

`jev rag trace` 可以查看本地运行目录，也可以读取随实验发布的 JSONL 轨迹索引。

```bash
TRACE=results/hotpotqa400_gate_traces.jsonl.gz
jev rag trace --input "$TRACE"
jev rag trace --input "$TRACE" --id "$QUESTION_ID"
jev rag trace --input "$TRACE" --id "$QUESTION_ID" --arm compact_80_80 --step 1
```

省略题号会列出可查的问题和方案。提供题号可查看整条轨迹，提供 `--step` 可定位单次模型调用。第 0 步是共享的初始草稿，第 1 步通常是首次充分性判断。分支列表记录每轮选择的动作和检索文档的索引。

公开索引含门控选项及概率、实际执行动作、检索位置、每次调用的 token 和耗时，以及请求和渲染提示的哈希。完整本地运行目录还可查看当时的状态、提示、生成输出和草稿。

```bash
jev rag trace --input runs/hotpotqa-compact --id "$QUESTION_ID" --step 1
```

## 复现 400 题门控策略

使用固定的 HotpotQA 题目清单和联合权重，按数据准备说明生成 `tasks.json`。三份配置分别设置全输入 95/80、全输入 80/80 和精简充分性 80/80。 [数据与权重准备](ITERATIVE_RAG.md)

```bash
for mode in 95_80 80_80 80_80_compact; do
  jev rag run --config "configs/hotpotqa_jev_$mode.json" \
    --tasks data/hotpotqa/tasks.json --manifest data/hotpotqa/split.json \
    --base "$BASE" --checkpoint "$JOINT_CKPT" --output "runs/hotpotqa-$mode"
  jev rag score --predictions "runs/hotpotqa-$mode/predictions.json" \
    --gold data/hotpotqa/gold.json --output "runs/hotpotqa-$mode/metrics.json"
done
```

使用自定义测试集和模型时，可按[接入示例](CUSTOM_INPUTS.md#用自己的模型跑-rag-并查询轨迹)生成本地运行目录，再用同样的命令查题号和步骤。
