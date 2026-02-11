from datasets import load_dataset
import json
import os
from collections import Counter

# Check available configs/splits
print("Checking dataset info...")
try:
    from datasets import get_dataset_config_names, get_dataset_split_names
    configs = get_dataset_config_names("liamdugan/raid")
    print(f"Available configs: {configs}")
    for config in configs:
        splits = get_dataset_split_names("liamdugan/raid", config)
        print(f"  Config '{config}' splits: {splits}")
except Exception as e:
    print(f"Error checking configs: {e}")

# Skip ahead in the dataset to find other domains
print("\nSkipping ahead to find more domains...")
ds = load_dataset("liamdugan/raid", split="train", streaming=True)

domain_counts = Counter()
model_counts = Counter()
total = 0
# Skip further ahead to find other domains
for i, item in enumerate(ds):
    if i < 200000:  # skip what we already saw
        if i % 100000 == 0:
            print(f"  Skipping {i}...")
        continue
    total += 1
    domain_counts[item['domain']] += 1
    model_counts[item['model']] += 1
    if total >= 200000:
        break
    if total % 50000 == 0:
        print(f"  Processed {total} new samples, domains so far: {list(domain_counts.keys())}")

print(f"\nSamples 200k-400k:")
print(f"  Domains: {dict(domain_counts)}")
print(f"  Models: {dict(model_counts)}")
