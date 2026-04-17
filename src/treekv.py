"""
TreeKV (Lian et al., 2025)
论文: https://arxiv.org/abs/2501.04987
IJCAI 2025

核心思想: 基于离散小波分析，对 KV Cache 实施 "sparse left, dense right" 的
         树形结构压缩。越旧的 token 块分配越少的预算（稀疏），越新的块保留越多
         细节（密集），本地窗口完全保留。

实现策略 — 块内 attention-based 选取 + 几何预算分配:
  1. 将历史区均分为 n_levels 个块（0=最旧，n_levels-1=最新历史块）
  2. 块 i 的预算 ∝ 2^i（几何级数），合计等于 k_ratio × hist_len
     → 最旧块：最小预算（约 total × 1/(2^n-1)）
     → 最新历史块：最大预算（约 total × 2^(n-1)/(2^n-1)）
  3. 每块内部用 SnapKV 同款的 attention importance score 选出 top-k 位置
  4. 本地窗口（local_window）完全保留

与 SnapKV 的关键区别:
  - SnapKV: 对整个历史做全局 top-k，不区分远近
  - TreeKV: 对每个历史块单独做局部 top-k，同时通过几何预算确保近处保留更多
  - 效果: 远处历史被更激进地压缩，近处历史几乎完整保留

优点:
  - 保留 token 原始 RoPE 位置编码（选取 token，不做池化），不引入混合位置误差
  - 预算分配对模型更友好：近处 token 对下一词预测更重要

本文件实现:
  - compress_kv_treekv: 树形 top-k 压缩 DynamicCache
  - compute_treekv_ppl: 滑动窗口 PPL（prefill → 压缩 → forward target）
  - benchmark_treekv_generation: 速度/显存 benchmark
"""
import os
import json
import time
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# src/ 目录的父目录是项目根，cache 在根目录下
CACHE_DIR  = os.path.join(os.path.dirname(__file__), "..", "cache")
ROOT_DIR   = os.path.join(os.path.dirname(__file__), "..")
MODEL_NAME = "EleutherAI/pythia-70m"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ── 模型加载 ───────────────────────────────────────────────────────────────────

def load_model():
    """
    使用 eager attention + fp32。
    TreeKV 需要 output_attentions=True（与 SnapKV 相同）。
    fp16 + eager 在 GPT-NeoX 上会产生 NaN，需用 fp32。
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        dtype=torch.float32,
        attn_implementation="eager",
    ).to(DEVICE).eval()
    return tokenizer, model


# ── TreeKV KV Cache 压缩 ───────────────────────────────────────────────────────

def compress_kv_treekv(
    past_kv,
    attentions,
    k_ratio: float  = 0.5,
    obs_window: int = 32,
    local_window: int = 32,
    n_levels: int   = 4,
):
    """
    TreeKV 树形 top-k 压缩。直接 in-place 修改 past_kv.layers[i].keys / .values。

    预算几何分配（n_levels=4 示例，denom=15）:
      块 0（最旧 25%）: budget = total × 1/15  ≈  6.7%
      块 1             : budget = total × 2/15  ≈ 13.3%
      块 2             : budget = total × 4/15  ≈ 26.7%
      块 3（最新历史）  : budget = total × 8/15  ≈ 53.3%
    local_window 完全保留。

    Args:
        past_kv:      DynamicCache，来自 prefill 阶段
        attentions:   list of [B, H, S_q, S_k] per layer（prefill attention）
        k_ratio:      历史区整体 KV 保留比例（总预算 = k_ratio × hist_len）
        obs_window:   用最后多少个 query token 观察 attention 分布
        local_window: 末尾完全保留的 token 数
        n_levels:     树的层数（决定几何分配的精细度）
    """
    for layer_idx, layer in enumerate(past_kv.layers):
        S = layer.keys.size(2)           # [B, H, S, D]
        if S <= local_window + 1:
            continue

        attn_w  = attentions[layer_idx]  # [B, H, S_q, S_k]
        S_q     = attn_w.size(2)
        obs_st  = max(0, S_q - obs_window)
        obs_attn = attn_w[:, :, obs_st:, :]    # [B, H, obs, S_k]

        # 等权聚合各 head 的 attention → 每个 key 位置的重要性
        importance = obs_attn.mean(dim=1).mean(dim=1)[0]   # [S_k]

        hist_len     = S - local_window
        chunk        = hist_len // n_levels
        if chunk == 0:
            continue

        # 几何预算总量
        total_budget = max(1, int(hist_len * k_ratio))
        denom        = (2 ** n_levels) - 1                 # 公比=2的等比数列之和

        selected_idx = []
        for i in range(n_levels):
            # 块 i：0=最旧，n_levels-1=最新历史块
            start  = i * chunk
            end    = (i + 1) * chunk if i < n_levels - 1 else hist_len
            actual = end - start

            # 几何预算：块 i 获得 total_budget × 2^i / denom
            block_budget = max(1, int(total_budget * (2 ** i) / denom))
            k_select     = min(actual, block_budget)

            # 块内局部 top-k（按 attention importance score）
            block_imp = importance[start:end]
            topk_local = block_imp.topk(k_select).indices + start   # 还原全局索引
            selected_idx.append(topk_local.sort().values)

        local_idx = torch.arange(S - local_window, S, device=DEVICE)
        all_idx   = torch.cat(selected_idx + [local_idx])           # 已按块顺序排列

        layer.keys   = layer.keys[:, :, all_idx, :]
        layer.values = layer.values[:, :, all_idx, :]


# ── 滑动窗口 PPL（带 TreeKV 压缩）────────────────────────────────────────────

def compute_treekv_ppl(
    model,
    input_ids: torch.Tensor,
    stride: int     = 512,
    max_length: int = 2048,
    k_ratio: float  = 0.5,
    obs_window: int = 32,
    local_window: int = 32,
    n_levels: int   = 4,
):
    """
    TreeKV PPL 评估：
    对每个 stride 大小的目标块，先做全量 prefill（获取 attention），
    用 TreeKV 几何预算压缩 KV Cache，再用压缩 KV forward 目标块，计算 NLL。
    """
    seq_len  = input_ids.size(1)
    nlls     = []
    loss_fn  = torch.nn.CrossEntropyLoss(reduction="sum")
    ctx_len  = max_length - stride

    for tgt_start in range(stride, seq_len, stride):
        tgt_end   = min(tgt_start + stride, seq_len)
        ctx_start = max(0, tgt_start - ctx_len)

        context = input_ids[:, ctx_start:tgt_start]
        target  = input_ids[:, tgt_start:tgt_end]

        if context.size(1) == 0:
            continue

        # ── Step 1: 全量 prefill，获取 attention weights ──
        with torch.no_grad():
            out = model(context, use_cache=True, output_attentions=True)

        past_kv    = out.past_key_values
        attentions = out.attentions

        # ── Step 2: TreeKV 几何预算压缩 ──
        compress_kv_treekv(
            past_kv, attentions,
            k_ratio=k_ratio, obs_window=obs_window,
            local_window=local_window, n_levels=n_levels,
        )

        # ── Step 3: 显式 position_ids，修正 RoPE 位置偏移 ──
        orig_ctx_len = context.size(1)
        tgt_len      = target.size(1)
        pos_ids      = torch.arange(
            orig_ctx_len, orig_ctx_len + tgt_len, device=DEVICE
        ).unsqueeze(0)

        with torch.no_grad():
            out2 = model(target, past_key_values=past_kv, use_cache=False,
                         position_ids=pos_ids)

        # ── Step 4: 计算 NLL ──
        ctx_last_logit = out.logits[:, -1:, :]
        full_logits    = torch.cat([ctx_last_logit, out2.logits[:, :-1, :]], dim=1)

        nll = loss_fn(
            full_logits.view(-1, full_logits.size(-1)),
            target.view(-1),
        )
        nlls.append(nll.item())

        if tgt_end == seq_len:
            break

    n_tokens = sum(
        min(s + stride, seq_len) - s
        for s in range(stride, seq_len, stride) if s < seq_len
    )
    nll_sum = sum(nlls)
    ppl     = np.exp(nll_sum / n_tokens)
    return ppl, nll_sum / n_tokens, n_tokens


# ── 速度 & 显存 benchmark ──────────────────────────────────────────────────────

def benchmark_treekv_generation(
    model, tokenizer, prompt: str,
    gen_len: int      = 200,
    k_ratio: float    = 0.5,
    obs_window: int   = 32,
    local_window: int = 32,
    n_levels: int     = 4,
):
    """
    TreeKV 生成 benchmark:
    1. 全量 prefill（获取 attention）
    2. TreeKV 几何预算压缩 KV Cache
    3. 在压缩 KV Cache 上做 greedy decode
    """
    input_ids  = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    prompt_len = input_ids.size(1)

    torch.cuda.reset_peak_memory_stats(DEVICE)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids, use_cache=True, output_attentions=True)
    past_kv    = out.past_key_values
    attentions = out.attentions
    compress_kv_treekv(past_kv, attentions,
                       k_ratio=k_ratio, obs_window=obs_window,
                       local_window=local_window, n_levels=n_levels)
    torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t0

    kv_len_after_compress = past_kv.get_seq_length()

    next_token = out.logits[:, -1:, :].argmax(-1)

    t1 = time.perf_counter()
    with torch.no_grad():
        for _ in range(gen_len - 1):
            out        = model(next_token, past_key_values=past_kv, use_cache=True)
            past_kv    = out.past_key_values
            next_token = out.logits[:, -1:, :].argmax(-1)
    torch.cuda.synchronize()
    t_decode = time.perf_counter() - t1

    peak_mem_mb = torch.cuda.max_memory_allocated(DEVICE) / 1024 / 1024

    return {
        "method"               : "treekv",
        "k_ratio"              : k_ratio,
        "obs_window"           : obs_window,
        "local_window"         : local_window,
        "n_levels"             : n_levels,
        "prompt_tokens"        : prompt_len,
        "generated_tokens"     : gen_len,
        "kv_len_after_compress": kv_len_after_compress,
        "prefill_sec"          : round(t_prefill, 4),
        "decode_sec"           : round(t_decode,  4),
        "decode_tps"           : round(gen_len / t_decode, 2),
        "peak_mem_mb"          : round(peak_mem_mb, 1),
    }


# ── wikitext PPL ───────────────────────────────────────────────────────────────

def eval_wikitext(model, tokenizer, max_length=2048, stride=512,
                  k_ratio=0.5, obs_window=32, local_window=32, n_levels=4):
    from datasets import load_dataset
    print("\n[wikitext-2-raw-v1] 加载测试集...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1",
                           cache_dir=CACHE_DIR, split="test")
    text = "\n\n".join(dataset["text"])

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
    print(f"  总 token 数: {input_ids.size(1):,}")
    print(f"  计算 PPL（max_length={max_length}, stride={stride}, "
          f"k_ratio={k_ratio}, obs={obs_window}, local={local_window}, "
          f"n_levels={n_levels}）...")

    ppl, nll_mean, n_tokens = compute_treekv_ppl(
        model, input_ids,
        stride=stride, max_length=max_length,
        k_ratio=k_ratio, obs_window=obs_window,
        local_window=local_window, n_levels=n_levels,
    )
    print(f"  [TreeKV wikitext-2] PPL = {ppl:.4f}  "
          f"(NLL={nll_mean:.4f}, tokens={n_tokens:,})")
    return ppl


def eval_pg19(model, tokenizer, max_length=2048, stride=512,
              k_ratio=0.5, obs_window=32, local_window=32, n_levels=4,
              pg19_tokens=8192, sample_idx=0):
    pg19_path = os.path.join(CACHE_DIR, "pg19_test_5samples.json")
    with open(pg19_path, encoding="utf-8") as f:
        samples = json.load(f)

    sample = samples[sample_idx]
    title  = sample.get("short_book_title", f"sample_{sample_idx}")
    text   = sample["text"]
    print(f"\n[pg-19] 样本: '{title}'")

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
    print(f"  总 token 数: {input_ids.size(1):,}，截取前 {pg19_tokens} tokens")

    input_ids = input_ids[:, :pg19_tokens]
    ppl, nll_mean, n_tokens = compute_treekv_ppl(
        model, input_ids,
        stride=stride, max_length=max_length,
        k_ratio=k_ratio, obs_window=obs_window,
        local_window=local_window, n_levels=n_levels,
    )
    print(f"  [TreeKV pg-19] PPL = {ppl:.4f}  "
          f"(NLL={nll_mean:.4f}, tokens={n_tokens:,})")
    return ppl


# ── 主程序 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TreeKV evaluation for Pythia-70M")
    parser.add_argument("--max_length",   type=int,   default=2048)
    parser.add_argument("--stride",       type=int,   default=512)
    parser.add_argument("--k_ratio",      type=float, default=0.5,
                        help="历史区整体 KV 保留比例（与 SnapKV 相同以公平对比）")
    parser.add_argument("--obs_window",   type=int,   default=32)
    parser.add_argument("--local_window", type=int,   default=32)
    parser.add_argument("--n_levels",     type=int,   default=4,
                        help="树的层数（n_levels=4 → 几何预算比例 1:2:4:8）")
    parser.add_argument("--pg19_tokens",  type=int,   default=8192)
    parser.add_argument("--gen_len",      type=int,   default=200)
    parser.add_argument("--skip_ppl",   action="store_true")
    parser.add_argument("--skip_bench", action="store_true")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print("加载模型（eager attention + fp32）...")
    tokenizer, model = load_model()
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    results = {}

    if not args.skip_ppl:
        wt_ppl = eval_wikitext(
            model, tokenizer,
            max_length=args.max_length, stride=args.stride,
            k_ratio=args.k_ratio, obs_window=args.obs_window,
            local_window=args.local_window, n_levels=args.n_levels,
        )
        pg_ppl = eval_pg19(
            model, tokenizer,
            max_length=args.max_length, stride=args.stride,
            k_ratio=args.k_ratio, obs_window=args.obs_window,
            local_window=args.local_window, n_levels=args.n_levels,
            pg19_tokens=args.pg19_tokens,
        )
        results["wikitext2_ppl"] = round(wt_ppl, 4)
        results["pg19_ppl"]      = round(pg_ppl, 4)

    if not args.skip_bench:
        prompt = ("In the beginning of the long story, the protagonist found himself "
                  "standing at the crossroads of fate. ")
        print(f"\n[Benchmark] prompt={len(tokenizer(prompt).input_ids)} tokens, "
              f"gen={args.gen_len}, k_ratio={args.k_ratio}, n_levels={args.n_levels}")
        bench = benchmark_treekv_generation(
            model, tokenizer, prompt,
            gen_len=args.gen_len,
            k_ratio=args.k_ratio, obs_window=args.obs_window,
            local_window=args.local_window, n_levels=args.n_levels,
        )
        for k, v in bench.items():
            print(f"  {k}: {v}")
        results["benchmark"] = bench

    out_path = os.path.join(ROOT_DIR, "results_treekv.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存至 {out_path}")


if __name__ == "__main__":
    main()
