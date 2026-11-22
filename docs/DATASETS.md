# 数据卡

| 数据 | Train 题 / 状态 | Validation 题 / 状态 | Test 题 / 状态 |
| --- | ---: | ---: | ---: |
| SufficiencyBench v1.1 | 1,352 / 2,704 | 264 / 528 | 1,069 / 2,138 |
| RAG-Necessity-987 | 697 / 1,822 | 140 / 370 | 150 / 396 |

## SufficiencyBench v1.1

沿用官方划分和充分性标签。一个问题可以对应多份检索上下文，评测按状态计分。目标是判断可见文档能否完整回答问题。上游文本适用各来源数据的条款，详情见 [SufficiencyBench](https://github.com/inab-lab/SufficiencyBench)。

## RAG-Necessity-987

由 HotpotQA 问题上的检索轨迹构成。GLM 与 DeepSeek 教师对候选追问打标签。三类分别表示值得执行、已有证据覆盖、对原问题无帮助。不确定标注从监督样本剔除。先取最早完成的 987 道题，再按题分组，固定 seed 42。标注可用性可能影响样本构成。评测反映模型对这些教师标签的拟合。

## 文件格式

`train.json` 每条含 `id`、`task_id`、`state`、固定选项与标签。模型输入由 `state` 和固定判断问题组成。训练标签只进入损失。验证和测试各用 `requests.json` 存输入，`gold.json` 按 ID 存标签。两个任务分别核对题目 ID，三个划分均无交集。源码版本控制默认忽略正文数据；公开发行时需核对来源和再分发条款。
