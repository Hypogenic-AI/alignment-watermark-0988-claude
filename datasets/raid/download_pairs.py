"""
Download RAID dataset samples focusing on base vs chat model pairs.
These pairs are critical for studying alignment watermarks.
"""
from datasets import load_dataset
import json
import os
from collections import Counter, defaultdict

output_dir = "/workspaces/alignment-watermark-0988-claude/datasets/raid"

# Target: collect samples for base/chat pairs
TARGET_MODELS = {
    'mistral', 'mistral-chat',
    'mpt', 'mpt-chat',
    'cohere', 'cohere-chat',
    'llama-chat',
    'gpt2',
    'chatgpt', 'gpt3', 'gpt4',
    'human',
}

TARGET_PER_MODEL = 500
MAX_SCAN = 800000

print("Loading RAID dataset (streaming)...")
ds = load_dataset("liamdugan/raid", split="train", streaming=True)

model_samples = defaultdict(list)
model_counts = Counter()
all_model_counts = Counter()
all_domain_counts = Counter()

for i, item in enumerate(ds):
    if i >= MAX_SCAN:
        break
    
    model = item['model']
    all_model_counts[model] += 1
    all_domain_counts[item['domain']] += 1
    
    # Focus on clean (non-attacked) samples
    if item['attack'] != 'none':
        continue
    
    if model in TARGET_MODELS and len(model_samples[model]) < TARGET_PER_MODEL:
        model_samples[model].append(item)
        model_counts[model] += 1
    
    if i % 100000 == 0:
        collected = sum(len(v) for v in model_samples.values())
        print(f"  Scanned {i}, collected {collected} samples across {len(model_samples)} models")
    
    # Early exit: check if ALL target models that exist have enough samples
    if len(model_samples) >= len(TARGET_MODELS):
        if all(len(model_samples[m]) >= TARGET_PER_MODEL for m in TARGET_MODELS):
            print(f"  All target models have {TARGET_PER_MODEL} samples at row {i}")
            break

print(f"\nTotal scanned: {i+1}")
print(f"\n=== ALL MODELS IN DATASET (from {i+1} rows) ===")
for model, count in sorted(all_model_counts.items(), key=lambda x: -x[1]):
    print(f"  {model}: {count}")

print(f"\n=== ALL DOMAINS IN DATASET ===")
for domain, count in sorted(all_domain_counts.items(), key=lambda x: -x[1]):
    print(f"  {domain}: {count}")

print(f"\n=== COLLECTED SAMPLES (attack='none' only) ===")
for model, samples in sorted(model_samples.items()):
    domains = Counter(s['domain'] for s in samples)
    decodings = Counter(str(s['decoding']) for s in samples)
    print(f"  {model}: {len(samples)} samples | domains: {dict(domains)} | decodings: {dict(decodings)}")

# Save all collected samples
all_samples = []
for model, samples in model_samples.items():
    all_samples.extend(samples)

print(f"\nTotal collected samples: {len(all_samples)}")

with open(os.path.join(output_dir, "raid_clean_samples.json"), "w") as f:
    json.dump(all_samples, f, indent=2, default=str)
print(f"Saved to {output_dir}/raid_clean_samples.json")

# Save model pairs specifically
pairs = {
    "mistral": {"base": [], "chat": []},
    "mpt": {"base": [], "chat": []},
    "cohere": {"base": [], "chat": []},
}

for sample in all_samples:
    model = sample['model']
    if model == 'mistral':
        pairs['mistral']['base'].append(sample)
    elif model == 'mistral-chat':
        pairs['mistral']['chat'].append(sample)
    elif model == 'mpt':
        pairs['mpt']['base'].append(sample)
    elif model == 'mpt-chat':
        pairs['mpt']['chat'].append(sample)
    elif model == 'cohere':
        pairs['cohere']['base'].append(sample)
    elif model == 'cohere-chat':
        pairs['cohere']['chat'].append(sample)

for family, data in pairs.items():
    print(f"\n  {family} pair: base={len(data['base'])}, chat={len(data['chat'])}")
    fname = os.path.join(output_dir, f"pair_{family}.json")
    with open(fname, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"    Saved to {fname}")

# Save summary
summary = {
    "total_scanned": i+1,
    "total_collected": len(all_samples),
    "all_models_in_dataset": dict(all_model_counts),
    "all_domains_in_dataset": dict(all_domain_counts),
    "collected_per_model": {m: len(s) for m, s in model_samples.items()},
    "base_chat_pairs": {
        "mistral / mistral-chat": {"base": len(pairs['mistral']['base']), "chat": len(pairs['mistral']['chat'])},
        "mpt / mpt-chat": {"base": len(pairs['mpt']['base']), "chat": len(pairs['mpt']['chat'])},
        "cohere / cohere-chat": {"base": len(pairs['cohere']['base']), "chat": len(pairs['cohere']['chat'])},
    },
    "filter": "attack='none' (clean, non-adversarial generations only)",
}
with open(os.path.join(output_dir, "dataset_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved dataset_summary.json")
