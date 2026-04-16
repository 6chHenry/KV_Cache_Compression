"""
SnapKV (Li et al., 2024)
论文: https://arxiv.org/abs/2404.14469

核心思想: 在 Prefill 阶段，用最后 obs_window 个 query token
         对全序列 key 做 attention pooling，选出重要的 top-k KV 位置，
         保留这些位置 + 最近 local_window 个位置，丢弃其余 KV Cache。
这样在长 prompt + 短 generation 场景下可显著节省显存和 decode 时间。

本文件实现:
  - compress_kv_snapkv: 根据 attention 权重压缩 DynamicCache
  - compute_snapkv_ppl: 滑动窗口 PPL（每窗口先 SnapKV 压缩再算 loss）
  - benchmark_snapkv_generation: 速度/显存 benchmark
"""
import os
import json
import time
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

CACHE_DIR  = os.path.join(os.path.dirname(__file__), "cache")
MODEL_NAME = "EleutherAI/pythia-70m"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ── 模型加载 ───────────────────────────────────────────────────────────────────

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        dtype=torch.float32,           # fp16 eager 会产生 NaN，用 fp32
        attn_implementation="eager",   # 必须使用 eager 以获取 attention weights
    ).to(DEVICE).eval()
    return tokenizer, model


# ── SnapKV KV Cache 压缩 ───────────────────────────────────────────────────────

def compress_kv_snapkv(
    past_kv,
    attentions,
    k_ratio: float = 0.5,
    obs_window: int = 32,
    local_window: int = 32,
):
    """
    SnapKV: 根据 prefill attention 权重压缩 KV Cache。
    直接 in-place 修改 past_kv.layers[i].keys / .values。

    Args:
        past_kv:      DynamicCache，来自 prefill 阶段
        attentions:   list of [B, H, S_q, S_k] per layer（prefill attention）
        k_ratio:      保留的 token 比例（相对于 S - local_window 部分）
        obs_window:   用最后多少个 query token 观察 attention 分布
        local_window: 始终保留末尾多少个 token（无论重要性）
    """
    for layer_idx, layer in enumerate(past_kv.layers):
        S = layer.keys.size(2)          # [B, H, S, D]

        # 如果序列很短，不压缩
        if S <= local_window + 1:
            continue

        attn_w = attentions[layer_idx]  # [B, H, S_q, S_k]
        S_q    = attn_w.size(2)

        # 取最后 obs_window 个 query 的 attention（对 prefill 阶段末尾部分）
        obs_start = max(0, S_q - obs_window)
        obs_attn  = attn_w[:, :, obs_start:, :]   # [B, H, obs, S_k]

        # 对 head 维度和 query 维度做平均 → 每个 key 位置的重要性分数
        importance = obs_attn.mean(dim=1).mean(dim=1)   # [B, S_k]
        importance = importance[0]                       # [S_k]

        # 末尾 local_window 个 token 直接保留，不参与选取
        candidate_len = S - local_window
        if candidate_len <= 0:
            continue

        # 从候选区域（非末尾部分）选 top-k
        k_select = max(1, int(candidate_len * k_ratio))
        cand_imp = importance[:candidate_len]
        topk_idx = cand_imp.topk(k_select).indices.sort().values   # 保持顺序

        local_idx = torch.arange(S - local_window, S, device=DEVICE)
        sel_idx   = torch.cat([topk_idx, local_idx])               # [k_select + local_window]

        # 按选出的索引截取 keys / values
        layer.keys   = layer.keys[:, :, sel_idx, :]
        layer.values = layer.values[:, :, sel_idx, :]


# ── 滑动窗口 PPL（带 SnapKV 压缩）────────────────────────────────────────────

def compute_snapkv_ppl(
    model,
    input_ids: torch.Tensor,
    stride: int   = 512,
    max_length: int = 2048,
    k_ratio: float = 0.5,
    obs_window: int = 32,
    local_window: int = 32,
):
    """
    SnapKV PPL 评估：
    对每个 stride 大小的目标块，先用 max_length-stride 长的上下文做全量 prefill，
    用 SnapKV 压缩 KV Cache，再用压缩后的 KV 对目标块做 forward，计算 NLL。
    这样测量的是"在压缩 KV 上真实的预测损失"，与 baseline 可对比。

    Args:
        input_ids:    [1, seq_len]，已在 DEVICE 上
        stride:       每次评估的目标 token 数
        max_length:   prefill 上下文长度上限
        k_ratio:      候选区域 KV 保留比例
        obs_window:   观察的末尾 query 数量
        local_window: 末尾始终保留的 token 数
    Returns:
        (ppl, nll_mean, n_tokens)
    """
    seq_len  = input_ids.size(1)
    nlls     = []
    loss_fn  = torch.nn.CrossEntropyLoss(reduction="sum")
    ctx_len  = max_length - stride    # prefill 上下文长度

    # 从 stride 处开始，每次评估 stride 个 token
    for tgt_start in range(stride, seq_len, stride):
        tgt_end    = min(tgt_start + stride, seq_len)
        ctx_start  = max(0, tgt_start - ctx_len)

        context = input_ids[:, ctx_start:tgt_start]     # 上下文
        target  = input_ids[:, tgt_start:tgt_end]       # 目标（要预测的 token）

        if context.size(1) == 0:
            continue

        # ── Step 1: 全量 prefill 上下文，获取 attention ──
        with torch.no_grad():
            out = model(context, use_cache=True, output_attentions=True)

        past_kv    = out.past_key_values
        attentions = out.attentions

        # ── Step 2: SnapKV 压缩 KV Cache ──
        compress_kv_snapkv(
            past_kv, attentions,
            k_ratio=k_ratio,
            obs_window=obs_window,
            local_window=local_window,
        )

        # ── Step 3: 用压缩 KV 对目标 token 做 forward，计算 NLL ──
        # SnapKV 压缩后 past_kv 变短，模型会错误推断 position_ids。
        # 必须显式传入正确位置：target token 紧接在原始上下文末尾。
        orig_ctx_len = context.size(1)
        tgt_len      = target.size(1)
        pos_ids      = torch.arange(
            orig_ctx_len, orig_ctx_len + tgt_len, device=DEVICE
        ).unsqueeze(0)   # [1, tgt_len]

        with torch.no_grad():
            out2 = model(target, past_key_values=past_kv, use_cache=False,
                         position_ids=pos_ids)

        logits  = out2.logits                         # [1, tgt_len, vocab]
        # 预测 target[1:] 的 NLL（target[0] 的预测已由上下文最后 logit 给出）
        # 这里计算 target 内部的自回归 NLL，连同第一个 token（由上下文预测）
        # 拼接：上下文最后 logit 预测 target[0]，target[:-1] 的 logit 预测 target[1:]
        ctx_last_logit = out.logits[:, -1:, :]        # [1, 1, vocab]
        full_logits    = torch.cat([ctx_last_logit, logits[:, :-1, :]], dim=1)  # [1, tgt_len, vocab]

        nll = loss_fn(
            full_logits.view(-1, full_logits.size(-1)),
            target.view(-1),
        )
        nlls.append(nll.item())

        if tgt_end == seq_len:
            break

    n_tokens = sum(
        min(tgt_start + stride, seq_len) - tgt_start
        for tgt_start in range(stride, seq_len, stride)
        if tgt_start < seq_len
    )
    nll_sum  = sum(nlls)
    ppl      = np.exp(nll_sum / n_tokens)
    return ppl, nll_sum / n_tokens, n_tokens


# ── 速度 & 显存 benchmark ──────────────────────────────────────────────────────

def benchmark_snapkv_generation(
    model, tokenizer, prompt: str,
    gen_len: int   = 200,
    k_ratio: float = 0.5,
    obs_window: int = 32,
    local_window: int = 32,
):
    """
    SnapKV 生成 benchmark:
    1. 完整 prefill（获取 attention）
    2. SnapKV 压缩 KV Cache
    3. 在压缩 KV Cache 上做 greedy decode
    """
    input_ids  = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    prompt_len = input_ids.size(1)

    torch.cuda.reset_peak_memory_stats(DEVICE)
    torch.cuda.synchronize()

    # ── prefill + SnapKV 压缩 ──
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids, use_cache=True, output_attentions=True)
    past_kv    = out.past_key_values
    attentions = out.attentions
    compress_kv_snapkv(past_kv, attentions,
                       k_ratio=k_ratio,
                       obs_window=obs_window,
                       local_window=local_window)
    torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t0

    kv_len_after_compress = past_kv.get_seq_length()

    # ── decode（在压缩 KV 上进行，不再裁剪）──
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
        "method"               : "snapkv",
        "k_ratio"              : k_ratio,
        "obs_window"           : obs_window,
        "local_window"         : local_window,
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
                  k_ratio=0.5, obs_window=32, local_window=32):
    from datasets import load_dataset
    print("\n[wikitext-2-raw-v1] 加载测试集...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1",
                           cache_dir=CACHE_DIR, split="test")
    text = "\n\n".join(dataset["text"])

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
    print(f"  总 token 数: {input_ids.size(1):,}")

    print(f"  计算 PPL（max_length={max_length}, stride={stride}, "
          f"k_ratio={k_ratio}, obs={obs_window}, local={local_window}）...")
    ppl, nll_mean, n_tokens = compute_snapkv_ppl(
        model, input_ids,
        stride=stride, max_length=max_length,
        k_ratio=k_ratio, obs_window=obs_window, local_window=local_window,
    )
    print(f"  [SnapKV wikitext-2] PPL = {ppl:.4f}  "
          f"(NLL={nll_mean:.4f}, tokens={n_tokens:,})")
    return ppl


def eval_pg19(model, tokenizer, max_length=2048, stride=512,
              k_ratio=0.5, obs_window=32, local_window=32,
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
    ppl, nll_mean, n_tokens = compute_snapkv_ppl(
        model, input_ids,
        stride=stride, max_length=max_length,
        k_ratio=k_ratio, obs_window=obs_window, local_window=local_window,
    )
    print(f"  [SnapKV pg-19] PPL = {ppl:.4f}  "
          f"(NLL={nll_mean:.4f}, tokens={n_tokens:,})")
    return ppl


# ── 主程序 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SnapKV evaluation for Pythia-70M")
    parser.add_argument("--max_length",   type=int,   default=2048,
                        help="滑动窗口大小")
    parser.add_argument("--stride",       type=int,   default=512,
                        help="滑动步长")
    parser.add_argument("--k_ratio",      type=float, default=0.5,
                        help="候选区域 KV 保留比例")
    parser.add_argument("--obs_window",   type=int,   default=32,
                        help="观察的末尾 query 数量")
    parser.add_argument("--local_window", type=int,   default=32,
                        help="末尾始终保留的 token 数")
    parser.add_argument("--pg19_tokens",  type=int,   default=8192,
                        help="pg-19 截取 token 数")
    parser.add_argument("--gen_len",      type=int,   default=200,
                        help="Benchmark 生成长度")
    parser.add_argument("--skip_ppl",   action="store_true")
    parser.add_argument("--skip_bench", action="store_true")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print("加载模型（eager attention）...")
    tokenizer, model = load_model()
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    results = {}

    if not args.skip_ppl:
        wt_ppl = eval_wikitext(model, tokenizer,
                               max_length=args.max_length, stride=args.stride,
                               k_ratio=args.k_ratio, obs_window=args.obs_window,
                               local_window=args.local_window)
        pg_ppl = eval_pg19(model, tokenizer,
                           max_length=args.max_length, stride=args.stride,
                           k_ratio=args.k_ratio, obs_window=args.obs_window,
                           local_window=args.local_window,
                           pg19_tokens=args.pg19_tokens)
        results["wikitext2_ppl"] = round(wt_ppl, 4)
        results["pg19_ppl"]      = round(pg_ppl, 4)

    if not args.skip_bench:
        prompt = ("In the beginning of the long story, the protagonist found himself "
                  "standing at the crossroads of fate. ")
        print(f"\n[Benchmark] prompt={len(tokenizer(prompt).input_ids)} tokens, "
              f"gen={args.gen_len}, k_ratio={args.k_ratio}")
        bench = benchmark_snapkv_generation(
            model, tokenizer, prompt,
            gen_len=args.gen_len, k_ratio=args.k_ratio,
            obs_window=args.obs_window, local_window=args.local_window,
        )
        for k, v in bench.items():
            print(f"  {k}: {v}")
        results["benchmark"] = bench

    out_path = os.path.join(os.path.dirname(__file__), "results_snapkv.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存至 {out_path}")


if __name__ == "__main__":
    main()
