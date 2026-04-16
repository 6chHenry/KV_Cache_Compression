"""
下载并缓存模型与数据集到本地 ./cache 目录
"""
import os

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ── 1. Pythia-70M ──────────────────────────────────────────────────────────────
print("=" * 60)
print("[1/3] 下载 Pythia-70M 模型...")
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "EleutherAI/pythia-70m"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
print(f"  参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
print("  Pythia-70M 下载完成 ✓")

# ── 2. wikitext-2-raw-v1 ───────────────────────────────────────────────────────
print("=" * 60)
print("[2/3] 下载 wikitext-2-raw-v1 数据集...")
from datasets import load_dataset

wikitext = load_dataset(
    "wikitext",
    "wikitext-2-raw-v1",
    cache_dir=CACHE_DIR,
)
test_tokens = sum(len(s) for s in wikitext["test"]["text"])
print(f"  test 集样本数: {len(wikitext['test'])}, 总字符数: {test_tokens:,}")
print("  wikitext-2-raw-v1 下载完成 ✓")

# ── 3. pg-19（只取 test split 前 5 个样本）─────────────────────────────────────
print("=" * 60)
print("[3/3] 下载 pg-19 数据集（streaming，取 test 前 5 条）...")

pg19_stream = load_dataset(
    "emozilla/pg19",
    split="test",
    streaming=True,
    cache_dir=CACHE_DIR,
    trust_remote_code=True,
)

samples = []
for i, sample in enumerate(pg19_stream):
    samples.append(sample)
    print(f"  sample {i}: title='{sample.get('short_book_title', 'N/A')}', "
          f"chars={len(sample['text']):,}")
    if i >= 4:
        break

print(f"  已缓存 {len(samples)} 条 pg-19 样本 ✓")

# 把这 5 条保存到本地 json，方便后续离线使用
import json
pg19_path = os.path.join(CACHE_DIR, "pg19_test_5samples.json")
with open(pg19_path, "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)
print(f"  已保存至 {pg19_path}")

print("=" * 60)
print("全部准备完毕！cache 目录：", CACHE_DIR)
