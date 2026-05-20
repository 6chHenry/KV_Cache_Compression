"""Download pg-19 test samples only (skip model/wikitext)."""
import json
import os
from datasets import load_dataset

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
NUM = int(os.environ.get("PG19_NUM", "20"))

print(f"Downloading pg-19 test split (first {NUM} samples)...")
stream = load_dataset("emozilla/pg19", split="test", streaming=True, trust_remote_code=True)
samples = []
for i, sample in enumerate(stream):
    samples.append(sample)
    if i >= NUM - 1:
        break

out = os.path.join(CACHE_DIR, f"pg19_test_{NUM}samples.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)
print(f"Saved {len(samples)} samples -> {out}")

legacy = os.path.join(CACHE_DIR, "pg19_test_5samples.json")
with open(legacy, "w", encoding="utf-8") as f:
    json.dump(samples[:5], f, ensure_ascii=False, indent=2)
