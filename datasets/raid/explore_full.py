from datasets import load_dataset
import json
import os
from collections import Counter

# Load the RAID dataset with streaming - go through more samples to see all domains
print("Loading RAID dataset (streaming)...")
ds = load_dataset("liamdugan/raid", split="train", streaming=True)

model_counts = Counter()
domain_counts = Counter()
decoding_counts = Counter()
attack_counts = Counter()
model_domain_pairs = Counter()
samples_by_model = {}

for i, item in enumerate(ds):
    if i >= 200000:
        break
    model = item['model']
    domain = item['domain']
    model_counts[model] += 1
    domain_counts[domain] += 1
    decoding_counts[str(item['decoding'])] += 1
    attack_counts[item['attack']] += 1
    model_domain_pairs[(model, domain)] += 1
    
    # Keep a few samples per model
    if model not in samples_by_model:
        samples_by_model[model] = []
    if len(samples_by_model[model]) < 3:
        samples_by_model[model].append(item)
    
    if i % 50000 == 0:
        print(f"  Processed {i} samples, {len(model_counts)} models, {len(domain_counts)} domains...")

print(f"\nTotal samples processed: {i+1}")
print(f"\n=== MODELS ({len(model_counts)}) ===")
for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
    print(f"  {model}: {count}")

print(f"\n=== DOMAINS ({len(domain_counts)}) ===")
for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
    print(f"  {domain}: {count}")

print(f"\n=== DECODINGS ===")
for dec, count in sorted(decoding_counts.items(), key=lambda x: -x[1]):
    print(f"  {dec}: {count}")

print(f"\n=== ATTACKS ===")
for atk, count in sorted(attack_counts.items(), key=lambda x: -x[1]):
    print(f"  {atk}: {count}")

# Identify base/chat pairs
print("\n=== BASE vs CHAT MODEL PAIRS ===")
all_models = set(model_counts.keys())
base_models = sorted([m for m in all_models if 'chat' not in m.lower() and m != 'human'])
chat_models = sorted([m for m in all_models if 'chat' in m.lower()])
print(f"Base models: {base_models}")
print(f"Chat models: {chat_models}")

# Show model x domain matrix
print("\n=== MODEL x DOMAIN MATRIX ===")
all_domains = sorted(domain_counts.keys())
print(f"{'Model':<20} " + " ".join(f"{d:<12}" for d in all_domains))
for model in sorted(model_counts.keys()):
    counts = [str(model_domain_pairs.get((model, d), 0)) for d in all_domains]
    print(f"{model:<20} " + " ".join(f"{c:<12}" for c in counts))

# Save detailed model info
output_dir = "/workspaces/alignment-watermark-0988-claude/datasets/raid"
with open(os.path.join(output_dir, "model_summary.json"), "w") as f:
    json.dump({
        "model_counts": dict(model_counts),
        "domain_counts": dict(domain_counts),
        "decoding_counts": dict(decoding_counts),
        "attack_counts": dict(attack_counts),
        "base_models": base_models,
        "chat_models": chat_models,
    }, f, indent=2)

# Save samples per model
with open(os.path.join(output_dir, "samples_by_model.json"), "w") as f:
    json.dump(samples_by_model, f, indent=2, default=str)

print("\nSaved model_summary.json and samples_by_model.json")
