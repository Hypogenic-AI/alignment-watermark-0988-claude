"""
Download samples from the 'extra' split which has different domains:
code, german, czech
"""
from datasets import load_dataset
import json
import os
from collections import Counter, defaultdict

output_dir = "/workspaces/alignment-watermark-0988-claude/datasets/raid"

TARGET_MODELS = {
    'mistral', 'mistral-chat',
    'mpt', 'mpt-chat',
    'llama-chat',
    'gpt2',
    'human',
}

TARGET_PER_MODEL_DOMAIN = 100
MAX_SCAN = 200000

print("Loading RAID extra split (streaming)...")
ds = load_dataset("liamdugan/raid", split="extra", streaming=True)

model_domain_samples = defaultdict(lambda: defaultdict(list))
all_model_counts = Counter()
all_domain_counts = Counter()
attack_counts = Counter()

for i, item in enumerate(ds):
    if i >= MAX_SCAN:
        break
    
    model = item['model']
    domain = item['domain']
    all_model_counts[model] += 1
    all_domain_counts[domain] += 1
    attack_counts[item['attack']] += 1
    
    if item['attack'] != 'none':
        continue
    
    if model in TARGET_MODELS and len(model_domain_samples[model][domain]) < TARGET_PER_MODEL_DOMAIN:
        model_domain_samples[model][domain].append(item)
    
    if i % 50000 == 0:
        total = sum(len(s) for md in model_domain_samples.values() for s in md.values())
        print(f"  Scanned {i}, collected {total} samples")

print(f"\nTotal scanned: {i+1}")
print(f"\n=== EXTRA SPLIT STATS ===")
print(f"Models: {dict(all_model_counts)}")
print(f"Domains: {dict(all_domain_counts)}")
print(f"Attacks: {dict(attack_counts)}")

print(f"\n=== COLLECTED SAMPLES ===")
all_extra = []
for model in sorted(model_domain_samples.keys()):
    for domain in sorted(model_domain_samples[model].keys()):
        samples = model_domain_samples[model][domain]
        all_extra.append({"model": model, "domain": domain, "count": len(samples)})
        print(f"  {model} / {domain}: {len(samples)}")

# Save extra domain samples
extra_samples = []
for model, domains in model_domain_samples.items():
    for domain, samples in domains.items():
        extra_samples.extend(samples)

with open(os.path.join(output_dir, "raid_extra_samples.json"), "w") as f:
    json.dump(extra_samples, f, indent=2, default=str)
print(f"\nSaved {len(extra_samples)} extra samples to raid_extra_samples.json")
