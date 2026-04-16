# 任务规划：语言模型高效推理（个人部分）

## 总体目标

在 Pythia-70M 上复现 StreamingLLM 和 SnapKV，并尝试改进，在 pg-19、wikitext 数据集上进行 PPL 和加速测试，最终提交公开 GitHub 仓库。

---

## Phase 0：前期准备

### 0.1 创建 GitHub 仓库
- 仓库名建议：`llm-efficient-inference` 或 `kv-cache-compression`
- 设为 Public
- 初始化 README.md 和 .gitignore（Python模板）
- 本地 `git clone` 并关联远程仓库

### 0.2 配置 Python 环境
```bash
conda create -n llm_accel python=3.10
conda activate llm_accel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # 按CUDA版本选择
pip install transformers datasets accelerate
pip install numpy pandas matplotlib tqdm
pip install sentencepiece  # tokenizer依赖
```

---

## Phase 1：模型与数据集准备

### 1.1 下载 Pythia-70M 模型
- **来源**：HuggingFace Model Hub
- **地址**：https://huggingface.co/EleutherAI/pythia-70m
- **方式**：使用 `transformers` 库自动下载并缓存

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "EleutherAI/pythia-70m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

- **注意**：模型约 160MB，需要梯度（`requires_grad=False`）以减少显存

### 1.2 下载 wikitext 数据集
- **来源**：HuggingFace Datasets Hub
- **地址**：https://huggingface.co/datasets/Salesforce/wikitext
- **版本**：`wikitext-2-raw-v1`（较小，适合快速测试）或 `wikitext-103-raw-v1`（更全面）
- **下载方式**：

```python
from datasets import load_dataset

# 推荐先用 wikitext-2-raw-v1 验证流程
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
test_data = dataset["test"]
```

### 1.3 下载 pg-19 数据集
- **来源**：HuggingFace Datasets Hub
- **地址**：https://huggingface.co/datasets/emozilla/pg19
- **特点**：超长文本（书籍），单条样本长度可达数万 token
- **下载方式**：

```python
from datasets import load_dataset

# pg-19 较大（约 10GB），按需只取部分
dataset = load_dataset("emozilla/pg19", split="test", streaming=True)
# 或仅取单个样本测试
sample = next(iter(dataset))
```

- **注意**：完整数据集约 10GB，建议使用 `streaming=True` 或只取少量样本（如1-5篇）

---

## Phase 2：Baseline 实现

### 2.1 标准自回归推理（Full KV Cache）
- 实现 `baseline.py`：加载模型，全量 KV Cache 推理
- 输入：长文本序列（来自 wikitext / pg-19）

### 2.2 PPL（困惑度）测量
- 公式：`PPL = exp(average cross-entropy loss)`
- 在 wikitext-2 test set 上，以固定窗口（如2048 token）滑动测量
- 在 pg-19 取单一样本（建议取5000~20000 token片段）测量

```python
import torch
from torch.nn import CrossEntropyLoss

def compute_ppl(model, tokenizer, text, stride=512, max_length=2048):
    # sliding window PPL computation
    ...
```

### 2.3 推理速度 & 显存测量
- 测量指标：
  - **吞吐量**（tokens/sec）
  - **峰值显存**（MB/GB）
  - **首token延迟**（TTFT，可选）
- 工具：`torch.cuda.memory_allocated()`，`time.perf_counter()`

---

## Phase 3：算法实现

### 3.1 StreamingLLM 实现
- **论文**：[Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)
- **核心思想**：保留最开始的几个 Attention Sink token + 最近的滑动窗口 token，丢弃中间的 KV Cache
- **参考代码**：https://github.com/mit-han-lab/streaming-llm
- **实现要点**：
  1. 修改模型的 attention 计算，支持截断 KV Cache
  2. 保留前 `k_sink`（如4）个 token 的 KV Cache
  3. 维护一个大小为 `window_size`（如1024）的滑动窗口
  4. 更新 position embedding（RoPE）以适配稀疏 KV Cache

### 3.2 SnapKV 实现
- **论文**：[SnapKV: LLM Knows What You are Looking for Before Generation](https://arxiv.org/abs/2404.14469)
- **核心思想**：通过观察 Prefill 阶段最后几个 query token 的 attention 分布，选出重要的 KV Cache 并压缩
- **参考代码**：https://github.com/FasterDecoding/SnapKV
- **实现要点**：
  1. Prefill 阶段：用最后 `window_size`（如32）个 query token 对所有 key 做 attention pooling
  2. 按 attention score 选出 top-k 重要的 KV 位置
  3. Decode 阶段：只保留选出的 KV + 最近的滑动窗口

### 3.3 改进方向（选做）
以下是几个可以尝试的改进点（二选一或结合）：

**方案A：StreamingLLM + SnapKV 融合**
- Prefill 用 SnapKV 的重要性选择，Decode 用 StreamingLLM 的 sink+window 策略

**方案B：自适应窗口大小**
- 根据每层的 attention entropy 动态调整 KV 保留数量，entropy 低（注意力集中）的层保留更少

**方案C：分层策略**
- 浅层保留更多 KV（语法信息重要），深层保留更少（语义更稀疏）

可以复现TreeKV

---

## Phase 4：实验与评估

### 4.1 PPL 对比实验
| 方法 | wikitext-2 PPL | pg-19 PPL |
|------|---------------|-----------|
| Baseline (Full KV) | ? | ? |
| StreamingLLM | ? | ? |
| SnapKV | ? | ? |
| 改进方法 | ? | ? |

### 4.2 速度 & 显存对比
| 方法 | tokens/sec | 峰值显存 (MB) | KV Cache压缩率 |
|------|-----------|-------------|--------------|
| Baseline | ? | ? | 1x |
| StreamingLLM | ? | ? | ? |
| SnapKV | ? | ? | ? |
| 改进方法 | ? | ? | ? |

### 4.3 测试配置
- 设备：GPU（推荐 CUDA），或 CPU（速度会慢很多）
- 序列长度：1024 / 2048 / 4096 三档
- KV 保留比例：25% / 50% / 75%（对应 StreamingLLM window size 或 SnapKV top-k）

---

## Phase 5：GitHub 仓库整理

### 5.1 代码结构
```
llm-efficient-inference/
├── README.md
├── requirements.txt
├── src/
│   ├── baseline.py          # 标准推理
│   ├── streaming_llm.py     # StreamingLLM实现
│   ├── snapkv.py            # SnapKV实现
│   ├── improved.py          # 改进方法
│   └── utils.py             # PPL计算、计时等工具
├── experiments/
│   ├── run_ppl.py           # PPL测试脚本
│   └── run_speed.py         # 速度测试脚本
└── results/
    └── results.md           # 实验结果汇总
```

### 5.2 README 必须包含
1. **项目介绍**：实现了哪些方法，核心思路
2. **环境安装**：`pip install -r requirements.txt`
3. **如何运行**：
   - 下载数据集的命令
   - 运行 baseline 的命令
   - 运行各优化方法的命令
4. **实验结果**：表格展示 PPL + 速度对比，并附简短分析
5. **参考文献**：引用原始论文

---

## 时间安排建议

| 阶段 | 预计工作量 | 说明 |
|------|-----------|------|
| Phase 0-1 环境+数据准备 | 0.5天 | 主要是下载和配置 |
| Phase 2 Baseline | 0.5天 | 实现PPL计算和基线测试 |
| Phase 3.1 StreamingLLM | 1-2天 | 需要理解并修改attention |
| Phase 3.2 SnapKV | 1-2天 | 需要理解prefill阶段attention分析 |
| Phase 3.3 改进 | 1天 | 基于前两个方法做组合或调整 |
| Phase 4 实验 | 1天 | 跑实验、记录结果 |
| Phase 5 整理仓库 | 0.5天 | 写README、整理代码 |

---

## 关键资源链接

| 资源 | 链接 |
|------|------|
| Pythia-70M 模型 | https://huggingface.co/EleutherAI/pythia-70m |
| KVPress 参考库 | https://github.com/NVIDIA/kvpress |
| StreamingLLM 论文 | https://arxiv.org/abs/2309.17453 |
| StreamingLLM 代码 | https://github.com/mit-han-lab/streaming-llm |
| SnapKV 论文 | https://arxiv.org/abs/2404.14469 |
| SnapKV 代码 | https://github.com/FasterDecoding/SnapKV |
| wikitext 数据集 | https://huggingface.co/datasets/Salesforce/wikitext |
| pg-19 数据集 | https://huggingface.co/datasets/emozilla/pg19 |
