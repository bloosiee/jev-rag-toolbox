# 仓库约定

- 一个命令入口，独立配置保存数据路径、训练配方和选参条件。布局参考 [LAMDA-PILOT](https://github.com/LAMDA-CL/LAMDA-PILOT) 的入口与实验配置。
- 同一任务配置可以切换评测方法。固定请求与标签分别存储，每次生成预测和指标文件。配置式评测参考 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)。
- 数据来源、标签、划分与限制放在数据卡。模型记录版本、输入协议、训练方法、指标与校验值。文档结构参考 [Hugging Face 数据卡](https://huggingface.co/docs/hub/datasets-cards) 与 [模型卡](https://huggingface.co/docs/hub/model-cards)。
- 冻结输入和检查点的 SHA-256，使用新目录记录实验。展开迭代流程时，把决策、行动与状态更新写成可分别检查的记录。
