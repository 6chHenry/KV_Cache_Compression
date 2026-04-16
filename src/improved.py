"""
KV Cache 压缩改进方法（对比实验）

在 SnapKV 基础上提出三种改进，统一评估并对比：

  方法 A — SnapKV + Sink 保护 (snapkv_sink)
    强制保留序列最前面的 k_sink 个 Attention Sink token，
    剩余预算从 [k_sink, S-local_window) 按 attention score 选 top-k。
    动机：sink token 在 SnapKV 纯排名中可能被丢弃，
         但 StreamingLLM 证明它们对模型输出有特殊稳定作用。

  方法 B — SnapKV + Adaptive Per-Head Budget (snapkv_adaptive)
    对每个 head 单独计算 importance，再以 attention 熵为权重加权聚合，
    高熵 head（注意力分散，需要更多 KV）在位置选取时话语权更大。
    总 KV 数量不变，但分配更合理。

  方法 C — SnapKV + Sink + Adaptive (snapkv_sink_adaptive)
    结合 A 和 B：sink token 强制保留 + 中间区域 entropy-weighted 选取。

所有方法 k_ratio=0.5（总 KV 数与标准 SnapKV 相同），可公平对比。
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
        dtype=torch.float32,
        attn_implementation="eager",
    ).to(DEVICE).eval()
    return tokenizer, model


# ── 公共工具：计算 entropy-weighted importance ─────────────────────────────────

def _entropy_weighted_importance(obs_attn):
    """
    给定 obs_attn [B, H, obs, S_k]，返回 entropy-weighted importance [S_k]。

    1. 对每个 head 取 obs 平均 → head_imp [H, S_k]
    2. 对每个 head 算 softmax entropy → entropy [H]
    3. 归一化 entropy 为权重，加权聚合 → importance [S_k]
    """
    head_imp  = obs_attn.mean(dim=2)[0]                              # [H, S_k]
    head_dist = head_imp.softmax(dim=-1)                             # [H, S_k]
    entropy   = -(head_dist * (head_dist + 1e-9).log()).sum(dim=-1)  # [H]
    weights   = entropy / (entropy.sum() + 1e-9)                     # [H]
    return (head_imp * weights.unsqueeze(-1)).sum(dim=0)             # [S_k]


def _equal_weight_importance(obs_attn):
    """标准 SnapKV：对 head 和 obs 等权平均 → importance [S_k]。"""
    return obs_attn.mean(dim=1).mean(dim=1)[0]   # [S_k]


# ── 方法 A：SnapKV + Sink 保护 ─────────────────────────────────────────────────

def compress_kv_sink(
    past_kv, attentions,
    k_sink: int = 4, k_ratio: float = 0.5,
    obs_window: int = 32, local_window: int = 32,
):
    """
    强制保留前 k_sink 个 token（Attention Sink），
    剩余预算从 [k_sink, S-local_window) 等权选 top-k。
    """
    for layer_idx, layer in enumerate(past_kv.layers):
        S = layer.keys.size(2)
        if S <= k_sink + local_window + 1:
            continue

        obs_start  = max(0, layer.keys.size(2) - obs_window)
        attn_w     = attentions[layer_idx]
        S_q        = attn_w.size(2)
        obs_start  = max(0, S_q - obs_window)
        obs_attn   = attn_w[:, :, obs_start:, :]

        importance = _equal_weight_importance(obs_attn)   # [S_k]

        sink_idx  = torch.arange(0, k_sink, device=DEVICE)
        mid_end   = S - local_window
        if mid_end > k_sink:
            mid_len  = mid_end - k_sink
            k_select = max(1, int(mid_len * k_ratio))
            topk_mid = importance[k_sink:mid_end].topk(k_select).indices + k_sink
            topk_mid = topk_mid.sort().values
        else:
            topk_mid = torch.empty(0, dtype=torch.long, device=DEVICE)

        local_idx = torch.arange(S - local_window, S, device=DEVICE)
        sel_idx   = torch.cat([sink_idx, topk_mid, local_idx])

        layer.keys   = layer.keys[:, :, sel_idx, :]
        layer.values = layer.values[:, :, sel_idx, :]


# ── 方法 B：SnapKV + Adaptive Per-Head Budget ──────────────────────────────────

def compress_kv_adaptive(
    past_kv, attentions,
    k_ratio: float = 0.5,
    obs_window: int = 32, local_window: int = 32,
):
    """
    Entropy-weighted importance 聚合：
    高熵 head 在位置选取时权重更大，总 KV 数不变。
    """
    for layer_idx, layer in enumerate(past_kv.layers):
        S = layer.keys.size(2)
        if S <= local_window + 1:
            continue

        attn_w    = attentions[layer_idx]
        S_q       = attn_w.size(2)
        obs_start = max(0, S_q - obs_window)
        obs_attn  = attn_w[:, :, obs_start:, :]

        importance    = _entropy_weighted_importance(obs_attn)   # [S_k]
        candidate_len = S - local_window
        k_select      = max(1, int(candidate_len * k_ratio))
        topk_idx      = importance[:candidate_len].topk(k_select).indices.sort().values

        local_idx = torch.arange(S - local_window, S, device=DEVICE)
        sel_idx   = torch.cat([topk_idx, local_idx])

        layer.keys   = layer.keys[:, :, sel_idx, :]
        layer.values = layer.values[:, :, sel_idx, :]


# ── 方法 C：SnapKV + Sink + Adaptive ──────────────────────────────────────────

def compress_kv_sink_adaptive(
    past_kv, attentions,
    k_sink: int = 4, k_ratio: float = 0.5,
    obs_window: int = 32, local_window: int = 32,
):
    """
    Sink 强制保护 + Entropy-Weighted 中间区域选取的组合。
    """
    for layer_idx, layer in enumerate(past_kv.layers):
        S = layer.keys.size(2)
        if S <= k_sink + local_window + 1:
            continue

        attn_w    = attentions[layer_idx]
        S_q       = attn_w.size(2)
        obs_start = max(0, S_q - obs_window)
        obs_attn  = attn_w[:, :, obs_start:, :]

        importance = _entropy_weighted_importance(obs_attn)   # [S_k]

        sink_idx = torch.arange(0, k_sink, device=DEVICE)
        mid_end  = S - local_window
        if mid_end > k_sink:
            mid_len  = mid_end - k_sink
            k_select = max(1, int(mid_len * k_ratio))
            topk_mid = importance[k_sink:mid_end].topk(k_select).indices + k_sink
            topk_mid = topk_mid.sort().values
        else:
            topk_mid = torch.empty(0, dtype=torch.long, device=DEVICE)

        local_idx = torch.arange(S - local_window, S, device=DEVICE)
        sel_idx   = torch.cat([sink_idx, topk_mid, local_idx])

        layer.keys   = layer.keys[:, :, sel_idx, :]
        layer.values = layer.values[:, :, sel_idx, :]


# ── 通用 PPL 计算（传入 compress_fn）────────────────────────────────────────

def compute_ppl_with_compress(
    model, input_ids: torch.Tensor,
    compress_fn,
    stride: int = 512, max_length: int = 2048,
):
    """
    滑动窗口 PPL，compress_fn 负责压缩 past_kv。
    compress_fn(past_kv, attentions) → None（in-place）
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

        with torch.no_grad():
            out = model(context, use_cache=True, output_attentions=True)
        past_kv    = out.past_key_values
        attentions = out.attentions

        compress_fn(past_kv, attentions)

        orig_ctx_len = context.size(1)
        tgt_len      = target.size(1)
        pos_ids      = torch.arange(
            orig_ctx_len, orig_ctx_len + tgt_len, device=DEVICE
        ).unsqueeze(0)

        with torch.no_grad():
            out2 = model(target, past_key_values=past_kv, use_cache=False,
                         position_ids=pos_ids)

        ctx_last_logit = out.logits[:, -1:, :]
        full_logits    = torch.cat([ctx_last_logit, out2.logits[:, :-1, :]], dim=1)

        nll = loss_fn(full_logits.view(-1, full_logits.size(-1)), target.view(-1))
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


# ── 通用 benchmark（传入 compress_fn）───────────────────────────────────────

def benchmark_with_compress(model, tokenizer, prompt, compress_fn, gen_len=200):
    input_ids  = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    prompt_len = input_ids.size(1)

    torch.cuda.reset_peak_memory_stats(DEVICE)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids, use_cache=True, output_attentions=True)
    compress_fn(out.past_key_values, out.attentions)
    torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t0

    past_kv    = out.past_key_values
    next_token = out.logits[:, -1:, :].argmax(-1)

    t1 = time.perf_counter()
    with torch.no_grad():
        for _ in range(gen_len - 1):
            out        = model(next_token, past_key_values=past_kv, use_cache=True)
            past_kv    = out.past_key_values
            next_token = out.logits[:, -1:, :].argmax(-1)
    torch.cuda.synchronize()
    t_decode = time.perf_counter() - t1

    return {
        "prompt_tokens"        : prompt_len,
        "generated_tokens"     : gen_len,
        "kv_len_after_compress": out.past_key_values.get_seq_length(),
        "prefill_sec"          : round(t_prefill, 4),
        "decode_sec"           : round(t_decode,  4),
        "decode_tps"           : round(gen_len / t_decode, 2),
        "peak_mem_mb"          : round(
            torch.cuda.max_memory_allocated(DEVICE) / 1024 / 1024, 1),
    }


# ── 数据集评估 ────────────────────────────────────────────────────────────────

def eval_wikitext(model, tokenizer, compress_fn, max_length=2048, stride=512):
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1",
                           cache_dir=CACHE_DIR, split="test")
    text      = "\n\n".join(dataset["text"])
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
    ppl, nll, n = compute_ppl_with_compress(
        model, input_ids, compress_fn,
        stride=stride, max_length=max_length,
    )
    return ppl, nll, n


def eval_pg19(model, tokenizer, compress_fn, max_length=2048, stride=512,
              pg19_tokens=8192, sample_idx=0):
    pg19_path = os.path.join(CACHE_DIR, "pg19_test_5samples.json")
    with open(pg19_path, encoding="utf-8") as f:
        samples = json.load(f)
    text      = samples[sample_idx]["text"]
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
    input_ids = input_ids[:, :pg19_tokens]
    ppl, nll, n = compute_ppl_with_compress(
        model, input_ids, compress_fn,
        stride=stride, max_length=max_length,
    )
    return ppl, nll, n


# ── 主程序：跑所有方法并汇总 ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="KV Cache 改进方法对比实验 — Pythia-70M"
    )
    parser.add_argument("--max_length",   type=int,   default=2048)
    parser.add_argument("--stride",       type=int,   default=512)
    parser.add_argument("--k_sink",       type=int,   default=4)
    parser.add_argument("--k_ratio",      type=float, default=0.5)
    parser.add_argument("--obs_window",   type=int,   default=32)
    parser.add_argument("--local_window", type=int,   default=32)
    parser.add_argument("--pg19_tokens",  type=int,   default=8192)
    parser.add_argument("--gen_len",      type=int,   default=200)
    parser.add_argument("--skip_ppl",   action="store_true")
    parser.add_argument("--skip_bench", action="store_true")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print("加载模型（eager attention）...")
    tokenizer, model = load_model()
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"超参：k_sink={args.k_sink}, k_ratio={args.k_ratio}, "
          f"obs={args.obs_window}, local={args.local_window}\n")

    # ── 定义三种方法 ──
    from functools import partial
    methods = {
        "snapkv_sink": partial(
            compress_kv_sink,
            k_sink=args.k_sink, k_ratio=args.k_ratio,
            obs_window=args.obs_window, local_window=args.local_window,
        ),
        "snapkv_adaptive": partial(
            compress_kv_adaptive,
            k_ratio=args.k_ratio,
            obs_window=args.obs_window, local_window=args.local_window,
        ),
        "snapkv_sink_adaptive": partial(
            compress_kv_sink_adaptive,
            k_sink=args.k_sink, k_ratio=args.k_ratio,
            obs_window=args.obs_window, local_window=args.local_window,
        ),
    }

    results = {}

    # ── PPL 评估 ──
    if not args.skip_ppl:
        print("=" * 60)
        print(f"{'方法':<25} {'wikitext-2 PPL':>15} {'pg-19 PPL':>12}")
        print("-" * 60)

        for name, compress_fn in methods.items():
            wt_ppl, wt_nll, wt_n = eval_wikitext(
                model, tokenizer, compress_fn,
                max_length=args.max_length, stride=args.stride,
            )
            pg_ppl, pg_nll, pg_n = eval_pg19(
                model, tokenizer, compress_fn,
                max_length=args.max_length, stride=args.stride,
                pg19_tokens=args.pg19_tokens,
            )
            print(f"  {name:<23} {wt_ppl:>15.4f} {pg_ppl:>12.4f}")
            results[name] = {
                "wikitext2_ppl": round(wt_ppl, 4),
                "pg19_ppl"     : round(pg_ppl, 4),
            }

        print("=" * 60)

    # ── Benchmark ──
    if not args.skip_bench:
        prompt = ("In the beginning of the long story, the protagonist found himself "
                  "standing at the crossroads of fate. ")
        print(f"\n[Benchmark] gen={args.gen_len} tokens")
        print(f"{'方法':<25} {'decode_tps':>12} {'peak_mem_mb':>13} {'kv_after':>10}")
        print("-" * 65)

        for name, compress_fn in methods.items():
            bench = benchmark_with_compress(
                model, tokenizer, prompt, compress_fn, gen_len=args.gen_len,
            )
            print(f"  {name:<23} {bench['decode_tps']:>12.1f} "
                  f"{bench['peak_mem_mb']:>13.1f} {bench['kv_len_after_compress']:>10}")
            results.setdefault(name, {})["benchmark"] = bench

        print("=" * 65)

    out_path = os.path.join(os.path.dirname(__file__), "results_improved.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存至 {out_path}")


if __name__ == "__main__":
    main()
