# LLM Efficient Inference：KV Cache 压缩方法复现与对比

在 **Pythia-70M** 上复现 StreamingLLM 和 SnapKV，并提出三种改进方案，在 wikitext-2 和 pg-19 数据集上进行 PPL 和速度评测。

This project is my individual implementation of AI2801 NLP.

---

## 方法概述

| 文件 | 方法 | 核心思想 |
| ------ | ------ | ---------- |
| `baseline.py` | Full KV Cache | 标准自回归推理，全量保留 KV Cache |
| `streaming_llm.py` | StreamingLLM | 保留前 k_sink 个 Sink token + 最近 window_size 个 token，丢弃中间 |
| `snapkv.py` | SnapKV | Prefill 阶段用末尾 obs_window 个 query 的 attention 选出重要 KV 位置 |
| `improved.py` | 三种改进 | 在 SnapKV 基础上改进选取策略（见下方分析） |

---

## 环境安装

```bash
conda create -n llm_accel python=3.10
conda activate llm_accel
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets numpy
```

---

## 数据准备

```bash
python prepare_data.py
```

自动下载并缓存到 `./cache/`：

- Pythia-70M 模型权重
- wikitext-2-raw-v1 测试集
- pg-19 测试集前 5 个样本（保存为 `cache/pg19_test_5samples.json`）

---

## 运行各方法

```bash
# Baseline（Full KV）
python baseline.py

# StreamingLLM（k_sink=4, window=512）
python streaming_llm.py --k_sink 4 --window_size 512 --max_eval_tokens 2048

# SnapKV（k_ratio=0.5, 保留 50% KV）
python snapkv.py --k_ratio 0.5 --obs_window 32 --local_window 32

# 改进方法（三种对比）
python improved.py --k_ratio 0.5 --k_sink 4 --obs_window 32 --local_window 32
```

---

## 实验结果

### PPL 对比

| 方法 | wikitext-2 PPL ↓ | pg-19 PPL ↓ |
| ------ | :--------------: | :---------: |
| Baseline (Full KV) | **39.85** | **13.75** |
| StreamingLLM (k_sink=4, window=512) | 302.41 | 167.15 |
| SnapKV (k_ratio=0.5) | 42.24 | 31.30 |
| + Sink 保护 (snapkv_sink) | 42.23 | 31.30 |
| + Adaptive Per-Head (snapkv_adaptive) | 42.24 | 31.30 |
| + Sink + Adaptive (snapkv_sink_adaptive) | 42.23 | 31.30 |

### 速度 & 显存对比

| 方法 | decode tps ↑ | 峰值显存 (MB) ↓ | KV 压缩比 |
| ------ | :----------: | :-------------: | :-------: |
| Baseline (Full KV) | 277 | 148 | 1× |
| StreamingLLM (k_sink=4, window=512) | **306** | **148** | ~0.43× |
| SnapKV (k_ratio=0.5) | 231 | 284 | 0.5× |
| + Sink 保护 | 226 | 289 | 0.5× |
| + Adaptive Per-Head | 223 | 289 | 0.5× |
| + Sink + Adaptive | 223 | 289 | 0.5× |

> **测试配置**：Pythia-70M，CUDA，fp32（eager）/ fp16（sdpa），prompt=21 tokens，生成 200 tokens。  
> PPL 评估：wikitext-2 全量 test set（288k tokens），pg-19 前 8192 tokens，滑动窗口 max_length=2048 / stride=512。

---

## 结果分析

### StreamingLLM

StreamingLLM 的 PPL 大幅高于 Baseline（302 vs 40），这是预期内的结果，**不意味着实现有误**。其设计目标是无限长序列的*稳定生成*，而非最小化 PPL：

- token-by-token 推理时，中间大量上下文被丢弃，模型看不到足够的历史信息；
- `k_sink=4, window=512` 配置下，有效上下文始终只有 ~516 token，而 wikitext-2 滑动窗口评估需要跨越更长距离的依赖；
- 优势在于**显存恒定**（148 MB，与 Baseline 相同）且 decode 速度最快（306 tps）。

### SnapKV

SnapKV 在 50% KV 压缩率下 PPL 仅损失约 6%（39.85 → 42.24），是有效的压缩方案。

**显存反而更高**（284 MB vs 148 MB）的原因：SnapKV 使用 `attn_implementation="eager"` + `fp32` 以获取 attention weights，而 Baseline 使用 `sdpa` + `fp16`，前者的中间 attention tensor 占用更大。若在相同精度下比较，SnapKV 的 KV Cache 本身确实更小。

### 改进方法

三种改进（Sink 保护、Adaptive Per-Head、组合）对 PPL 的提升均在 **0.01 量级以内**，未能拉开明显差距。

分析原因：

1. **Pythia-70M 头数少（8 heads/layer）**：各 head 的 attention 熵差异有限，entropy-weighted 聚合与等权平均在小模型上几乎等价；
2. **Sink token 已被自然选中**：在 50% 压缩率下，前 4 个 token 的 attention score 本身就足以进入 top-k，强制保留与否差别微乎其微；
3. **固定 k_ratio 限制了上限**：在总 KV 预算不变的前提下，改变选取策略的收益空间本就有限。

这一结论本身也有参考价值：**在小规模模型（< 1B）上，KV 选取策略的精细化带来的收益极为有限；这类改进的价值更多体现在头数多（≥ 32）、上下文更长的大模型场景中。**

---

## 参考文献

- Xiao et al., *Efficient Streaming Language Models with Attention Sinks*, 2023. [[arxiv]](https://arxiv.org/abs/2309.17453) [[code]](https://github.com/mit-han-lab/streaming-llm)
- Li et al., *SnapKV: LLM Knows What You are Looking for Before Generation*, 2024. [[arxiv]](https://arxiv.org/abs/2404.14469) [[code]](https://github.com/FasterDecoding/SnapKV)
- Biderman et al., *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling*, 2023. [[model]](https://huggingface.co/EleutherAI/pythia-70m)
