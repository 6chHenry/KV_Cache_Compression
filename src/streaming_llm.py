"""
StreamingLLM (Xiao et al., 2023)
论文: https://arxiv.org/abs/2309.17453

核心思想: 保留前 k_sink 个 Attention Sink token 的 KV Cache +
         最近 window_size 个 token 的 KV Cache，丢弃中间部分。
这样 KV Cache 大小恒定为 (k_sink + window_size)，支持无限长序列推理。

本文件实现:
  - trim_kv_streaming: 裁剪 DynamicCache
  - compute_streaming_ppl: 流式 token-by-token PPL 测量
  - benchmark_streaming_generation: 速度/显存 benchmark
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
        dtype=torch.float16,
        # sdpa（默认）：fp16 下稳定，StreamingLLM 无需 output_attentions
    ).to(DEVICE).eval()
    return tokenizer, model


# ── KV Cache 裁剪 ──────────────────────────────────────────────────────────────

def trim_kv_streaming(past_kv, k_sink: int, window_size: int):
    """
    将 DynamicCache 裁剪为 sink tokens + 最近 window_size tokens。
    直接修改 past_kv.layers[i].keys / .values（in-place）。

    Args:
        past_kv:     transformers DynamicCache 对象
        k_sink:      保留的 attention sink token 数量（序列开头）
        window_size: 保留的滑动窗口大小（序列末尾）
    """
    for layer in past_kv.layers:
        S = layer.keys.size(2)   # [B, H, S, D]
        if S <= k_sink + window_size:
            continue
        sink_k = layer.keys[:, :, :k_sink, :]
        sink_v = layer.values[:, :, :k_sink, :]
        win_k  = layer.keys[:, :, S - window_size:, :]
        win_v  = layer.values[:, :, S - window_size:, :]
        layer.keys   = torch.cat([sink_k, win_k], dim=2)
        layer.values = torch.cat([sink_v, win_v], dim=2)


# ── 流式 PPL（token-by-token）──────────────────────────────────────────────────

def compute_streaming_ppl(
    model,
    input_ids: torch.Tensor,
    k_sink: int = 4,
    window_size: int = 512,
    max_eval_tokens: int = 2048,
):
    """
    逐 token 推理，维护 sink+window KV Cache，计算 PPL。

    Args:
        input_ids:        [1, seq_len]，已在 DEVICE 上
        k_sink:           sink token 数量
        window_size:      滑动窗口大小
        max_eval_tokens:  最多评估多少 token（避免过慢）
    Returns:
        (ppl, nll_mean, n_tokens)
    """
    seq_len   = min(input_ids.size(1), max_eval_tokens + 1)
    input_ids = input_ids[:, :seq_len]

    nlls    = []
    past_kv = None

    # 用前 k_sink 个 token 做 prefill，建立 sink KV Cache
    prefix       = input_ids[:, :k_sink]
    with torch.no_grad():
        out     = model(prefix, use_cache=True)
    past_kv = out.past_key_values

    # 之后逐 token 滑动，每次喂一个 token，记录其预测 NLL
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    for t in range(k_sink, seq_len - 1):
        cur_tok  = input_ids[:, t:t+1]        # [1, 1]
        tgt_tok  = input_ids[:, t+1:t+2]      # [1, 1]

        with torch.no_grad():
            out = model(cur_tok, past_key_values=past_kv, use_cache=True)

        logits  = out.logits[:, -1, :].float()  # [1, vocab]，转 fp32 避免 fp16 溢出
        nll     = loss_fn(logits, tgt_tok.view(-1))
        nlls.append(nll.item())

        past_kv = out.past_key_values
        # 裁剪 KV Cache：保持 sink + window
        trim_kv_streaming(past_kv, k_sink, window_size)

    n_tokens = len(nlls)
    nll_mean = sum(nlls) / n_tokens
    ppl      = np.exp(nll_mean)
    return ppl, nll_mean, n_tokens


# ── 速度 & 显存 benchmark ──────────────────────────────────────────────────────

def benchmark_streaming_generation(
    model, tokenizer, prompt: str,
    gen_len: int  = 200,
    k_sink: int   = 4,
    window_size: int = 256,
):
    """
    使用 StreamingLLM（sink + window KV Cache）进行生成，测量速度和显存。
    """
    input_ids  = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    prompt_len = input_ids.size(1)

    torch.cuda.reset_peak_memory_stats(DEVICE)
    torch.cuda.synchronize()

    # ── prefill ──
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t0

    past_kv    = out.past_key_values
    next_token = out.logits[:, -1:, :].argmax(-1)

    # ── decode with streaming KV ──
    t1 = time.perf_counter()
    with torch.no_grad():
        for _ in range(gen_len - 1):
            # 每次 decode 后裁剪 KV Cache
            trim_kv_streaming(past_kv, k_sink, window_size)
            out        = model(next_token, past_key_values=past_kv, use_cache=True)
            past_kv    = out.past_key_values
            next_token = out.logits[:, -1:, :].argmax(-1)
    torch.cuda.synchronize()
    t_decode = time.perf_counter() - t1

    peak_mem_mb = torch.cuda.max_memory_allocated(DEVICE) / 1024 / 1024
    # 结束时 KV Cache 大小（token 数）
    final_kv_len = past_kv.get_seq_length()

    return {
        "method"          : "streaming_llm",
        "k_sink"          : k_sink,
        "window_size"     : window_size,
        "prompt_tokens"   : prompt_len,
        "generated_tokens": gen_len,
        "final_kv_len"    : final_kv_len,
        "prefill_sec"     : round(t_prefill, 4),
        "decode_sec"      : round(t_decode,  4),
        "decode_tps"      : round(gen_len / t_decode, 2),
        "peak_mem_mb"     : round(peak_mem_mb, 1),
    }


# ── wikitext PPL ───────────────────────────────────────────────────────────────

def eval_wikitext(model, tokenizer, k_sink=4, window_size=512,
                  max_eval_tokens=2048):
    from datasets import load_dataset
    print("\n[wikitext-2-raw-v1] 加载测试集...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1",
                           cache_dir=CACHE_DIR, split="test")
    text = "\n\n".join(dataset["text"])

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
    print(f"  总 token 数: {input_ids.size(1):,}，评估前 {max_eval_tokens} tokens")

    ppl, nll_mean, n_tokens = compute_streaming_ppl(
        model, input_ids, k_sink=k_sink, window_size=window_size,
        max_eval_tokens=max_eval_tokens,
    )
    print(f"  [StreamingLLM wikitext-2] PPL = {ppl:.4f}  "
          f"(NLL={nll_mean:.4f}, tokens={n_tokens:,}, "
          f"k_sink={k_sink}, window={window_size})")
    return ppl


def eval_pg19(model, tokenizer, k_sink=4, window_size=512,
              max_eval_tokens=2048, sample_idx=0):
    pg19_path = os.path.join(CACHE_DIR, "pg19_test_5samples.json")
    with open(pg19_path, encoding="utf-8") as f:
        samples = json.load(f)

    sample = samples[sample_idx]
    title  = sample.get("short_book_title", f"sample_{sample_idx}")
    text   = sample["text"]
    print(f"\n[pg-19] 样本: '{title}'")

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
    print(f"  总 token 数: {input_ids.size(1):,}，评估前 {max_eval_tokens} tokens")

    ppl, nll_mean, n_tokens = compute_streaming_ppl(
        model, input_ids, k_sink=k_sink, window_size=window_size,
        max_eval_tokens=max_eval_tokens,
    )
    print(f"  [StreamingLLM pg-19] PPL = {ppl:.4f}  "
          f"(NLL={nll_mean:.4f}, tokens={n_tokens:,}, "
          f"k_sink={k_sink}, window={window_size})")
    return ppl


# ── 主程序 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="StreamingLLM evaluation for Pythia-70M")
    parser.add_argument("--k_sink",          type=int, default=4,
                        help="Attention sink token 数量")
    parser.add_argument("--window_size",     type=int, default=512,
                        help="滑动窗口大小")
    parser.add_argument("--max_eval_tokens", type=int, default=2048,
                        help="PPL 评估最多 token 数（token-by-token 较慢）")
    parser.add_argument("--gen_len",         type=int, default=200,
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
                               k_sink=args.k_sink, window_size=args.window_size,
                               max_eval_tokens=args.max_eval_tokens)
        pg_ppl = eval_pg19(model, tokenizer,
                           k_sink=args.k_sink, window_size=args.window_size,
                           max_eval_tokens=args.max_eval_tokens)
        results["wikitext2_ppl"] = round(wt_ppl, 4)
        results["pg19_ppl"]      = round(pg_ppl, 4)

    if not args.skip_bench:
        prompt = ("In the beginning of the long story, the protagonist found himself "
                  "standing at the crossroads of fate. ")
        print(f"\n[Benchmark] prompt={len(tokenizer(prompt).input_ids)} tokens, "
              f"gen={args.gen_len}, k_sink={args.k_sink}, window={args.window_size}")
        bench = benchmark_streaming_generation(
            model, tokenizer, prompt,
            gen_len=args.gen_len, k_sink=args.k_sink, window_size=args.window_size,
        )
        for k, v in bench.items():
            print(f"  {k}: {v}")
        results["benchmark"] = bench

    out_path = os.path.join(os.path.dirname(__file__), "results_streaming_llm.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存至 {out_path}")


if __name__ == "__main__":
    main()
