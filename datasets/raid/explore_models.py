from datasets import load_dataset
import json
from collections import Counter

# Load the RAID dataset with streaming to count models
print("Loading RAID dataset (streaming)...")
ds = load_dataset("liamdugan/raid", split="train", streaming=True)

# Take a larger sample to see all models
model_counts = Counter()
domain_counts = Counter()
decoding_counts = Counter()
attack_counts = Counter()
samples_by_model = {}

for i, item in enumerate(ds):
    if i >= 50000:
        break
    model = item['model']
    model_counts[model] += 1
    domain_counts[item['domain']] += 1
    decoding_counts[str(item['decoding'])] += 1
    attack_counts[item['attack']] += 1
    
    # Keep a few samples per model
    if model not in samples_by_model:
        samples_by_model[model] = []
    if len(samples_by_model[model]) < 3:
        samples_by_model[model].append(item)
    
    if i % 10000 == 0:
        print(f"  Processed {i} samples, found {len(model_counts)} models so far...")

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
base_models = [m for m in all_models if 'chat' not in m.lower() and m != 'human']
chat_models = [m for m in all_models if 'chat' in m.lower()]
print(f"Base models: {sorted(base_models)}")
print(f"Chat models: {sorted(chat_models)}")

# Save detailed model info
output_dir = "/workspaces/alignment-watermark-0988-claude/datasets/raid"
with open(os.path.join(output_dir, "model_summary.json"), "w") as f:
    json.dump({
        "model_counts": dict(model_counts),
        "domain_counts": dict(domain_counts),
        "decoding_counts": dict(decoding_counts),
        "attack_counts": dict(attack_counts),
        "base_models": sorted(base_models),
        "chat_models": sorted(chat_models),
    }, f, indent=2)

# Save sample per model
with open(os.path.join(output_dir, "samples_by_model.json"), "w") as f:
    json.dump(samples_by_model, f, indent=2, default=str)

print("\nSaved model_summary.json and samples_by_model.json")
