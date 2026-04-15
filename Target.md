# 大作业：语言模型高效推理（个人部分）

- **算法实现：**
  - 在KVPress或者其他加速优化方法中选择并复现

- **Baseline设置**
  - 所有实验都应该在Pythia-70M模型上用无训练方法进行优化
  - 在pg-19, wikitext等数据集上进行ppl测试和加速测试
    注：pg-19为超长文本数据集，可取单一sample进行测试即可

- **创建个人独立的公开Github仓库：**
  - 包含你实现的全部代码和README
    建议使用git的相关功能进行代码历史跟踪

- **README至少需要包含：**
  - 如何运行你的代码
  - 一个简短报告展示加速/优化效果

- **评分标准：**
  - 根据所实现算法和提交的仓库可复现性进行评分

https://huggingface.co/EleutherAI/pythia-70m
https://github.com/NVIDIA/kvpress

## 项目计划

准备：Pythia-70M模型,pg-19,wikitext数据集，建立GitHub仓库

实现StreamingLLM,SnapKV,并做改进。
