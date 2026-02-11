from datasets import load_dataset
import json
import os

# Load the RAID dataset
print("Loading RAID dataset...")
ds = load_dataset("liamdugan/raid", split="train", streaming=True)

# Take first 1000 samples to examine structure
samples = []
for i, item in enumerate(ds):
    if i >= 1000:
        break
    samples.append(item)

print(f"Collected {len(samples)} samples")
print(f"Column names: {list(samples[0].keys())}")
print(f"First sample keys: {list(samples[0].keys())}")

# Save samples
output_dir = "/workspaces/alignment-watermark-0988-claude/datasets/raid"
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "samples.json"), "w") as f:
    json.dump(samples[:10], f, indent=2, default=str)

# Print some stats
models = set()
for s in samples:
    if 'model' in s:
        models.add(s['model'])
    elif 'source' in s:
        models.add(s['source'])

print(f"\nModels/sources found: {models}")
print(f"\nSample record:")
for k, v in samples[0].items():
    val_str = str(v)[:200]
    print(f"  {k}: {val_str}")
