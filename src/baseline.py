"""
Baseline: Pythia-70M 全量 KV Cache 推理
- PPL 测量（滑动窗口）
- 推理速度 & 峰值显存测量
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
        MODEL_NAME, cache_dir=CACHE_DIR, torch_dtype=torch.float16
    ).to(DEVICE).eval()
    return tokenizer, model


# ── PPL（滑动窗口）─────────────────────────────────────────────────────────────

def compute_ppl(model, input_ids: torch.Tensor, stride: int = 512, max_length: int = 2048):
    """
    滑动窗口 PPL。
    input_ids: shape [1, seq_len]，已在 DEVICE 上。
    返回 (ppl, nll_mean, n_tokens)。
    """
    seq_len  = input_ids.size(1)
    nlls     = []
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end        = min(begin + max_length, seq_len)
        target_len = end - prev_end          # 本窗口新增的 token 数
        input_ids_chunk = input_ids[:, begin:end]

        # 用 logits 手动计算新增部分的 NLL，避免 outputs.loss 对全窗口平均的偏差
        with torch.no_grad():
            logits = model(input_ids_chunk).logits   # [1, L, vocab]

        shift_l = logits[:, :-1, :].contiguous()
        shift_t = input_ids_chunk[:, 1:].contiguous()

        offset  = shift_l.size(1) - target_len
        loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
        nll     = loss_fn(
            shift_l[:, offset:, :].view(-1, shift_l.size(-1)),
            shift_t[:, offset:].view(-1),
        )
        nlls.append(nll.item())
        prev_end = end
        if end == seq_len:
            break

    nll_sum  = sum(nlls)
    n_tokens = seq_len - 1          # 预测的 token 总数（排除第一个）
    ppl      = np.exp(nll_sum / n_tokens)
    return ppl, nll_sum / n_tokens, n_tokens


# ── 速度 & 显存 benchmark ──────────────────────────────────────────────────────

def benchmark_generation(model, tokenizer, prompt: str, gen_len: int = 200):
    """
    测量 prefill 时间、decode 吞吐量（tokens/sec）、峰值显存（MB）。
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
    past_kv   = out.past_key_values

    # ── decode（greedy）──
    next_token = out.logits[:, -1, :].argmax(-1, keepdim=True)

    t1 = time.perf_counter()
    with torch.no_grad():
        for _ in range(gen_len - 1):
            out        = model(next_token, past_key_values=past_kv, use_cache=True)
            past_kv    = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(-1, keepdim=True)
    torch.cuda.synchronize()
    t_decode = time.perf_counter() - t1

    peak_mem_mb = torch.cuda.max_memory_allocated(DEVICE) / 1024 / 1024

    return {
        "prompt_tokens"   : prompt_len,
        "generated_tokens": gen_len,
        "prefill_sec"     : round(t_prefill, 4),
        "decode_sec"      : round(t_decode,  4),
        "decode_tps"      : round(gen_len / t_decode, 2),
        "peak_mem_mb"     : round(peak_mem_mb, 1),
    }


# ── wikitext PPL ───────────────────────────────────────────────────────────────

def eval_wikitext(model, tokenizer, max_length=2048, stride=512):
    from datasets import load_dataset
    print("\n[wikitext-2-raw-v1] 加载测试集...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1",
                           cache_dir=CACHE_DIR, split="test")
    text = "\n\n".join(dataset["text"])
    print(f"  总字符数: {len(text):,}")

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
    print(f"  总 token 数: {input_ids.size(1):,}")

    print(f"  计算 PPL（max_length={max_length}, stride={stride}）...")
    ppl, nll_mean, n_tokens = compute_ppl(model, input_ids,
                                          stride=stride, max_length=max_length)
    print(f"  [wikitext-2] PPL = {ppl:.4f}  (NLL={nll_mean:.4f}, tokens={n_tokens:,})")
    return ppl


# ── pg-19 PPL ─────────────────────────────────────────────────────────────────

def eval_pg19(model, tokenizer, max_length=2048, stride=512,
              pg19_tokens=8192, sample_idx=0):
    pg19_path = os.path.join(CACHE_DIR, "pg19_test_5samples.json")
    with open(pg19_path, encoding="utf-8") as f:
        samples = json.load(f)

    sample = samples[sample_idx]
    title  = sample.get("short_book_title", f"sample_{sample_idx}")
    text   = sample["text"]
    print(f"\n[pg-19] 样本: '{title}'  原文字符数: {len(text):,}")

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
    total_tok = input_ids.size(1)
    print(f"  总 token 数: {total_tok:,}，截取前 {pg19_tokens} tokens")

    input_ids = input_ids[:, :pg19_tokens]
    print(f"  计算 PPL（max_length={max_length}, stride={stride}）...")
    ppl, nll_mean, n_tokens = compute_ppl(model, input_ids,
                                          stride=stride, max_length=max_length)
    print(f"  [pg-19] PPL = {ppl:.4f}  (NLL={nll_mean:.4f}, tokens={n_tokens:,})")
    return ppl


# ── 主程序 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Baseline evaluation for Pythia-70M")
    parser.add_argument("--max_length",  type=int, default=2048,
                        help="滑动窗口大小")
    parser.add_argument("--stride",      type=int, default=512,
                        help="滑动步长")
    parser.add_argument("--pg19_tokens", type=int, default=8192,
                        help="pg-19 截取 token 数")
    parser.add_argument("--gen_len",     type=int, default=200,
                        help="generation benchmark 生成长度")
    parser.add_argument("--skip_ppl",   action="store_true")
    parser.add_argument("--skip_bench", action="store_true")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print("加载模型...")
    tokenizer, model = load_model()
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    results = {}

    # ── PPL ──
    if not args.skip_ppl:
        wt_ppl = eval_wikitext(model, tokenizer,
                               max_length=args.max_length, stride=args.stride)
        pg_ppl = eval_pg19(model, tokenizer,
                           max_length=args.max_length, stride=args.stride,
                           pg19_tokens=args.pg19_tokens)
        results["wikitext2_ppl"] = round(wt_ppl, 4)
        results["pg19_ppl"]      = round(pg_ppl, 4)

    # ── Speed & Memory ──
    if not args.skip_bench:
        prompt = ("In the beginning of the long story, the protagonist found himself "
                  "standing at the crossroads of fate. ")
        print(f"\n[Benchmark] prompt={len(tokenizer(prompt).input_ids)} tokens, "
              f"gen={args.gen_len} tokens")
        bench = benchmark_generation(model, tokenizer, prompt, gen_len=args.gen_len)
        for k, v in bench.items():
            print(f"  {k}: {v}")
        results["benchmark"] = bench

    # ── 保存结果 ──
    out_path = os.path.join(os.path.dirname(__file__), "results_baseline.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存至 {out_path}")


if __name__ == "__main__":
    main()