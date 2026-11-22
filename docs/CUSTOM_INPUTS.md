# 自定义日志、测试集和模型

## 决策日志

支持 JSON 数组、JSONL 和 CSV。用映射文件把标准字段名指向源字段，点号可定位嵌套字段。CSV 的文档、历史追问和问题队列列使用 JSON 字符串。字段名已匹配时省略 `--mapping`。

```bash
jev convert decisions --task sufficiency --input examples/custom_decisions.jsonl \
  --mapping examples/decision_mapping.json --labeled --output runs/custom-decisions

jev eval --task sufficiency --requests runs/custom-decisions/requests.json \
  --gold runs/custom-decisions/gold.json --method rule --output runs/custom-decision-eval
```

充分性输入 `original_question` 和 `retrieved_documents`。必要性再提供 `executed_followups`（每条含 `question` 和 `answer`）、`current_intermediate_answer`、`pending_questions` 和 `proposed_followup`。标注列默认叫 `label`，取值沿用两项任务的固定选项。没有标注时省略 `--labeled` 和 `--gold`，批量命令仍写预测。

## RAG 测试集

每行放 `id`、`question` 和 `documents`。文档提供 `title`，以及 `text` 或 `sentences`。带答案的记录加 `answer` 和 `--labeled`。

```bash
jev convert rag --input examples/custom_rag.jsonl --labeled --output runs/custom-rag
```

把生成的 `tasks.json` 交给 `jev rag run --tasks`，省略 HotpotQA 冻结清单。用单独生成的 `gold.json` 调用 `jev rag score`。自定义集沿用 HotpotQA 答案 EM/F1 口径。转换输出带 SHA-256 清单。

## 模型适配器

把本地 Python 模块加入 `PYTHONPATH`，按 `module:factory` 指定工厂。工厂接受一个配置字典，默认是空字典，也可通过 `--adapter-config` 提供 JSON 文件。仅加载信任的 Python 模块。认证信息从环境变量读取。

| 方法 | 输入 | 返回 |
|---|---|---|
| `predict(kind, request)` | `request` 包含状态、判断问题和选项 | `choice`，可附 `probabilities` 和 token 数 |
| `generate(prompt, *, max_new_tokens, schema=None)` | 提示、生成预算和可选 JSON Schema | `text`，可附 `truncated` 和 token 数 |

决策日志用 `jev predict --adapter` 或 `jev eval --adapter`。RAG 用 `--generator-adapter` 替换生成模型，用 `--gate-adapter` 替换门控。保留原生组件时仍提供基座与检查点。选择 Jev 路由配置时，门控适配器提供给定选项的归一化概率，由配置中的阈值决定动作。外部概率由适配器负责校准。返回 token 数时使用 `input_tokens` 和 `output_tokens`，未提供时汇总记为 `null`。

```bash
PYTHONPATH=examples/adapters jev rag run --config configs/hotpotqa_pipeline_jev.json \
  --tasks runs/custom-rag/tasks.json --generator-adapter sample_adapter:build \
  --gate-adapter sample_adapter:build --output runs/custom-rag-run

jev rag score --predictions runs/custom-rag-run/predictions.json \
  --gold runs/custom-rag/gold.json --output runs/custom-rag-run/metrics.json
```

示例适配器用固定回答演示接口。接真实模型时把两项方法换成模型调用即可。
