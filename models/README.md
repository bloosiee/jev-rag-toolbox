# 模型卡

| 模型 | 验证入选配置 | 独立测试准确率 | 原始 Mentat v0.3 准确率 | 主要指标 |
| --- | --- | ---: | ---: | ---: |
| Jev 充分性 | batch 16 / 2e-5 | 2,014 / 2,138（94.20%） | 1,689 / 2,138（79.00%） | AUROC 0.9809 |
| Jev 必要性 | batch 64 / 1e-5 | 337 / 396（85.10%） | 310 / 396（78.28%） | macro-F1 0.7962 |

表格来自已有独立训练实验，按决策状态计分。两项都是原生输入协议，端到端问答表现需要独立评测。推理默认温度为 1。生产门槛应在部署分布上的保留数据校准。

每套目录由 `adapter_config.json`、`adapter_model.safetensors`、`decision_readout.json`、`decision_readout.safetensors` 四个文件组成。对应的 SHA-256 在本目录。当训练新权重时，使用对应运行的检查点清单。

## 下载权重

三套模型都从相同的 Mentat v0.3 初始化。充分性和必要性各自单独训练；联合模型混合两项训练数据，选定的配置为 batch 16、学习率 2e-5，供全文迭代 RAG 示例使用。

| 权重包 | 用途 | SHA-256 文件清单 |
| --- | --- | --- |
| [sufficiency.tar.gz](https://github.com/bloosiee/jev-rag-toolbox/releases/download/v0.1.0-models/sufficiency.tar.gz) | 原生证据充分性判断 | [清单](sufficiency.sha256.json) |
| [necessity.tar.gz](https://github.com/bloosiee/jev-rag-toolbox/releases/download/v0.1.0-models/necessity.tar.gz) | 原生三分类追问必要性判断 | [清单](necessity.sha256.json) |
| [joint.tar.gz](https://github.com/bloosiee/jev-rag-toolbox/releases/download/v0.1.0-models/joint.tar.gz) | 联合模型，迭代 RAG 的充分性与必要性门控 | [清单](joint.sha256.json) |

每个压缩包包含对应的 LoRA、决策读出层、配置和许可。解压后，将权重目录传给 `--checkpoint`。基座从上游获取。 [Qwen3.5-4B 固定版本](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a) · [Mentat v0.3 初始化模型](https://huggingface.co/yunqu/mentat-sys1-v0.3)
