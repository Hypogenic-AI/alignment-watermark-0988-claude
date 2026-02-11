"""
Experiment 2: Zero-Shot Statistical Detection
Uses a local reference model (GPT-2) to compute per-token log-probabilities,
then applies detection signals (mean log-prob, log-rank, entropy) to distinguish
human vs. base vs. aligned text.
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "data")
PLOTS_DIR = os.path.join(BASE_DIR, "results", "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Use GPU 0 (most free memory)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 512  # Max tokens per text


def load_model(model_name="gpt2-large"):
    """Load reference model for computing log-probabilities."""
    print(f"Loading {model_name} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def compute_token_logprobs(text, model, tokenizer):
    """Compute per-token log-probabilities for a text using a reference model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    input_ids = inputs["input_ids"].to(DEVICE)

    if input_ids.shape[1] < 2:
        return None

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # Shift: predict token[i+1] from logits[i]
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    # Log-softmax
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

    # Per-token log-probabilities
    token_logprobs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Probabilities for rank calculation
    probs = torch.softmax(shift_logits, dim=-1)

    # Compute ranks of actual tokens
    sorted_probs, _ = probs.sort(dim=-1, descending=True)
    actual_probs = probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    ranks = (probs >= actual_probs.unsqueeze(-1)).sum(dim=-1).float()

    # Entropy
    entropy = -(probs * log_probs).sum(dim=-1)

    return {
        'logprobs': token_logprobs[0].cpu().numpy(),
        'ranks': ranks[0].cpu().numpy(),
        'entropy': entropy[0].cpu().numpy(),
        'num_tokens': shift_labels.shape[1],
    }


def compute_detection_features(token_data):
    """Compute detection metrics from per-token data."""
    if token_data is None:
        return None

    logprobs = token_data['logprobs']
    ranks = token_data['ranks']
    entropy = token_data['entropy']

    return {
        'mean_logprob': float(np.mean(logprobs)),
        'std_logprob': float(np.std(logprobs)),
        'mean_rank': float(np.mean(ranks)),
        'mean_log_rank': float(np.mean(np.log(ranks + 1))),
        'mean_entropy': float(np.mean(entropy)),
        'std_entropy': float(np.std(entropy)),
        'num_tokens': token_data['num_tokens'],
    }


def load_raid_data():
    """Load RAID samples organized by category."""
    data_path = os.path.join(BASE_DIR, "datasets", "raid", "raid_clean_samples.json")
    with open(data_path) as f:
        samples = json.load(f)

    by_model = {}
    for s in samples:
        model = s['model']
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(s)

    categories = {
        'human': by_model.get('human', []),
        'base_mistral': by_model.get('mistral', []),
        'aligned_mistral': by_model.get('mistral-chat', []),
        'base_mpt': by_model.get('mpt', []),
        'aligned_mpt': by_model.get('mpt-chat', []),
        'base_cohere': by_model.get('cohere', []),
        'aligned_cohere': by_model.get('cohere-chat', []),
        'chatgpt': by_model.get('chatgpt', []),
        'gpt4': by_model.get('gpt4', []),
    }
    return categories


def compute_auroc_with_ci(scores_positive, scores_negative, n_bootstrap=1000):
    """Compute AUROC with bootstrap confidence interval.
    positive = AI text (labeled 1), negative = human text (labeled 0)."""
    y_true = np.array([1] * len(scores_positive) + [0] * len(scores_negative))
    y_scores = np.concatenate([scores_positive, scores_negative])

    auroc = roc_auc_score(y_true, y_scores)

    # Bootstrap CI
    rng = np.random.RandomState(SEED)
    aurocs_boot = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        y_t = y_true[idx]
        y_s = y_scores[idx]
        if len(np.unique(y_t)) < 2:
            continue
        aurocs_boot.append(roc_auc_score(y_t, y_s))

    ci_low = np.percentile(aurocs_boot, 2.5)
    ci_high = np.percentile(aurocs_boot, 97.5)

    return auroc, ci_low, ci_high


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 2: Zero-Shot Statistical Detection")
    print("=" * 70)

    model, tokenizer = load_model("gpt2-large")
    categories = load_raid_data()

    # Compute detection features for all texts
    all_features = {}
    for cat_name, texts in categories.items():
        print(f"\nProcessing {cat_name} ({len(texts)} texts)...")
        feats = []
        for t in tqdm(texts, desc=cat_name):
            token_data = compute_token_logprobs(t['generation'], model, tokenizer)
            det_feats = compute_detection_features(token_data)
            if det_feats is not None:
                det_feats['category'] = cat_name
                feats.append(det_feats)
        all_features[cat_name] = feats
        print(f"  -> {len(feats)} valid texts processed")

    # Combine into DataFrame
    all_rows = []
    for cat_name, feats in all_features.items():
        all_rows.extend(feats)
    df = pd.DataFrame(all_rows)

    # Add alignment type
    def alignment_type(cat):
        if cat == 'human':
            return 'human'
        elif cat.startswith('base_'):
            return 'base'
        elif cat.startswith('aligned_') or cat in ('chatgpt', 'gpt4'):
            return 'aligned'
        return 'other'

    df['alignment'] = df['category'].apply(alignment_type)

    # Save raw features
    df.to_csv(os.path.join(RESULTS_DIR, "exp2_detection_features.csv"), index=False)

    # ---- DETECTION PERFORMANCE ----
    print("\n" + "=" * 70)
    print("DETECTION PERFORMANCE (AUROC)")
    print("=" * 70)

    human_feats = df[df['category'] == 'human']
    detection_metrics = ['mean_logprob', 'mean_log_rank', 'mean_entropy']

    # For detection: higher mean_logprob = more AI-like (model assigns high prob)
    # Higher mean_entropy = more human-like (more uncertain)
    # So for mean_logprob, AI score = mean_logprob
    # For mean_entropy, AI score = -mean_entropy (lower entropy = more AI)
    # For mean_log_rank, AI score = -mean_log_rank (lower rank = more AI)

    metric_signs = {
        'mean_logprob': 1,     # Higher = more AI-like
        'mean_log_rank': -1,   # Lower = more AI-like
        'mean_entropy': -1,    # Lower = more AI-like
    }

    auroc_results = []
    ai_categories = [c for c in categories.keys() if c != 'human']

    for cat in ai_categories:
        cat_df = df[df['category'] == cat]
        for metric in detection_metrics:
            sign = metric_signs[metric]
            ai_scores = (sign * cat_df[metric]).values
            human_scores = (sign * human_feats[metric]).values

            auroc, ci_low, ci_high = compute_auroc_with_ci(ai_scores, human_scores)
            # Ensure AUROC >= 0.5 (flip if needed)
            if auroc < 0.5:
                auroc = 1 - auroc
                ci_low, ci_high = 1 - ci_high, 1 - ci_low

            auroc_results.append({
                'category': cat,
                'alignment': alignment_type(cat),
                'metric': metric,
                'auroc': auroc,
                'ci_low': ci_low,
                'ci_high': ci_high,
            })
            print(f"  {cat:20s} | {metric:20s} | AUROC={auroc:.4f} [{ci_low:.4f}, {ci_high:.4f}]")

    auroc_df = pd.DataFrame(auroc_results)
    auroc_df.to_csv(os.path.join(RESULTS_DIR, "exp2_auroc_results.csv"), index=False)

    # ---- AGGREGATE: Base vs. Aligned ----
    print("\n" + "=" * 70)
    print("AGGREGATE: Base vs. Aligned Detection")
    print("=" * 70)

    families = ['mistral', 'mpt', 'cohere']
    aggregate_results = []

    for metric in detection_metrics:
        sign = metric_signs[metric]
        human_scores = (sign * human_feats[metric]).values

        for family in families:
            base_scores = (sign * df[df['category'] == f'base_{family}'][metric]).values
            aligned_scores = (sign * df[df['category'] == f'aligned_{family}'][metric]).values

            auroc_base, ci_low_b, ci_high_b = compute_auroc_with_ci(base_scores, human_scores)
            auroc_aligned, ci_low_a, ci_high_a = compute_auroc_with_ci(aligned_scores, human_scores)

            if auroc_base < 0.5:
                auroc_base = 1 - auroc_base
                ci_low_b, ci_high_b = 1 - ci_high_b, 1 - ci_low_b
            if auroc_aligned < 0.5:
                auroc_aligned = 1 - auroc_aligned
                ci_low_a, ci_high_a = 1 - ci_high_a, 1 - ci_low_a

            delta = auroc_aligned - auroc_base

            aggregate_results.append({
                'family': family,
                'metric': metric,
                'auroc_base': auroc_base,
                'auroc_aligned': auroc_aligned,
                'delta': delta,
                'ci_base': f"[{ci_low_b:.4f}, {ci_high_b:.4f}]",
                'ci_aligned': f"[{ci_low_a:.4f}, {ci_high_a:.4f}]",
            })

            print(f"  {family:10s} | {metric:20s} | Base={auroc_base:.4f}, Aligned={auroc_aligned:.4f}, "
                  f"Delta={delta:+.4f}")

    agg_df = pd.DataFrame(aggregate_results)
    agg_df.to_csv(os.path.join(RESULTS_DIR, "exp2_aggregate_auroc.csv"), index=False)

    # ---- VISUALIZATIONS ----
    print("\nGenerating plots...")

    # Plot 1: AUROC comparison (base vs aligned, grouped by family)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, metric in enumerate(detection_metrics):
        ax = axes[idx]
        metric_data = agg_df[agg_df['metric'] == metric]

        x = np.arange(len(families))
        width = 0.35
        bars1 = ax.bar(x - width/2, metric_data['auroc_base'].values, width,
                        label='Base', color='#3498db', alpha=0.85)
        bars2 = ax.bar(x + width/2, metric_data['auroc_aligned'].values, width,
                        label='Aligned', color='#e74c3c', alpha=0.85)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
        ax.set_xticks(x)
        ax.set_xticklabels([f.capitalize() for f in families])
        ax.set_ylabel('AUROC')
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_ylim(0.4, 1.0)
        ax.legend()

    plt.suptitle('Detection AUROC: Base vs Aligned Models (Human as Negative)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "exp2_auroc_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Detection feature distributions
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, metric in enumerate(detection_metrics):
        ax = axes[idx]
        for atype, color in [('human', '#2ecc71'), ('base', '#3498db'), ('aligned', '#e74c3c')]:
            vals = df[df['alignment'] == atype][metric].values
            ax.hist(vals, bins=50, alpha=0.5, color=color, label=atype, density=True)
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_xlabel(metric)
        ax.set_ylabel('Density')
        ax.legend()

    plt.suptitle('Detection Feature Distributions: Human vs Base vs Aligned', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "exp2_feature_distributions.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: ROC curves for best metric (mean_logprob) for each family
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    best_metric = 'mean_logprob'
    sign = metric_signs[best_metric]

    for idx, family in enumerate(families):
        ax = axes[idx]
        human_scores = (sign * human_feats[best_metric]).values

        for variant, color, label in [
            (f'base_{family}', '#3498db', 'Base'),
            (f'aligned_{family}', '#e74c3c', 'Aligned'),
        ]:
            ai_scores = (sign * df[df['category'] == variant][best_metric]).values
            y_true = np.array([1] * len(ai_scores) + [0] * len(human_scores))
            y_scores = np.concatenate([ai_scores, human_scores])
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            auroc = roc_auc_score(y_true, y_scores)
            if auroc < 0.5:
                fpr, tpr = 1 - fpr, 1 - tpr
                auroc = 1 - auroc
            ax.plot(fpr, tpr, color=color, label=f'{label} (AUROC={auroc:.3f})', linewidth=2)

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title(f'{family.capitalize()}')
        ax.legend(loc='lower right')

    plt.suptitle(f'ROC Curves: {best_metric} Detection', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "exp2_roc_curves.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    print(f"\nResults saved to {RESULTS_DIR}")
    print(f"Plots saved to {PLOTS_DIR}")
    print("Experiment 2 complete.")

    return df, auroc_df, agg_df


if __name__ == "__main__":
    run_experiment()
