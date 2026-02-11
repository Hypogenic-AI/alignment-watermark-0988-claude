"""
Experiment 1: Distributional Feature Analysis
Compares linguistic features of base vs. aligned vs. human text from RAID dataset.
"""
import json
import os
import sys
import re
import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
np.random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "data")
PLOTS_DIR = os.path.join(BASE_DIR, "results", "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def tokenize_simple(text):
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r'\b\w+\b', text.lower())


def compute_text_features(text):
    """Compute distributional features for a single text."""
    tokens = tokenize_simple(text)
    if len(tokens) == 0:
        return None

    # Word-level features
    types = set(tokens)
    ttr = len(types) / len(tokens)  # type-token ratio

    # N-gram diversity (distinct n-gram ratio)
    def distinct_ngram_ratio(toks, n):
        ngrams = [tuple(toks[i:i+n]) for i in range(len(toks) - n + 1)]
        if len(ngrams) == 0:
            return 0.0
        return len(set(ngrams)) / len(ngrams)

    dist1 = distinct_ngram_ratio(tokens, 1)
    dist2 = distinct_ngram_ratio(tokens, 2)
    dist3 = distinct_ngram_ratio(tokens, 3)

    # Sentence-level features
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sent_lengths = [len(tokenize_simple(s)) for s in sentences]

    mean_sent_len = np.mean(sent_lengths) if sent_lengths else 0
    std_sent_len = np.std(sent_lengths) if len(sent_lengths) > 1 else 0

    # Word length features
    word_lengths = [len(t) for t in tokens]
    mean_word_len = np.mean(word_lengths)

    # Vocabulary richness (Hapax legomena ratio)
    freq = Counter(tokens)
    hapax = sum(1 for w, c in freq.items() if c == 1)
    hapax_ratio = hapax / len(types) if types else 0

    # Text length
    num_tokens = len(tokens)
    num_chars = len(text)

    return {
        'num_tokens': num_tokens,
        'num_chars': num_chars,
        'num_sentences': len(sentences),
        'type_token_ratio': ttr,
        'distinct_1gram': dist1,
        'distinct_2gram': dist2,
        'distinct_3gram': dist3,
        'mean_sent_length': mean_sent_len,
        'std_sent_length': std_sent_len,
        'mean_word_length': mean_word_len,
        'hapax_ratio': hapax_ratio,
    }


def load_raid_data():
    """Load all RAID samples and organize by category."""
    data_path = os.path.join(BASE_DIR, "datasets", "raid", "raid_clean_samples.json")
    with open(data_path) as f:
        samples = json.load(f)

    # Organize by model
    by_model = {}
    for s in samples:
        model = s['model']
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(s)

    # Define categories
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


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 1: Distributional Feature Analysis")
    print("=" * 70)

    categories = load_raid_data()

    # Compute features for all texts
    all_features = {}
    for cat_name, texts in categories.items():
        print(f"Computing features for {cat_name} ({len(texts)} texts)...")
        feats = []
        for t in texts:
            f = compute_text_features(t['generation'])
            if f is not None:
                f['category'] = cat_name
                feats.append(f)
        all_features[cat_name] = feats
        print(f"  -> {len(feats)} valid texts")

    # Combine into DataFrame
    all_rows = []
    for cat_name, feats in all_features.items():
        all_rows.extend(feats)
    df = pd.DataFrame(all_rows)

    # Add alignment status column
    def alignment_type(cat):
        if cat == 'human':
            return 'human'
        elif cat.startswith('base_'):
            return 'base'
        elif cat.startswith('aligned_') or cat in ('chatgpt', 'gpt4'):
            return 'aligned'
        return 'other'

    df['alignment'] = df['category'].apply(alignment_type)

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS BY ALIGNMENT TYPE")
    print("=" * 70)
    feature_cols = ['type_token_ratio', 'distinct_1gram', 'distinct_2gram', 'distinct_3gram',
                    'mean_sent_length', 'std_sent_length', 'mean_word_length', 'hapax_ratio',
                    'num_tokens']

    summary = df.groupby('alignment')[feature_cols].agg(['mean', 'std'])
    print(summary.to_string())

    # Statistical tests: base vs aligned for each family
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS: Base vs. Aligned (per family)")
    print("=" * 70)

    families = ['mistral', 'mpt', 'cohere']
    test_results = []

    for family in families:
        base_df = df[df['category'] == f'base_{family}']
        aligned_df = df[df['category'] == f'aligned_{family}']

        print(f"\n--- {family.upper()} ---")
        for feat in feature_cols:
            base_vals = base_df[feat].values
            aligned_vals = aligned_df[feat].values

            # Mann-Whitney U test
            stat, p = stats.mannwhitneyu(base_vals, aligned_vals, alternative='two-sided')

            # Cohen's d
            pooled_std = np.sqrt((np.std(base_vals)**2 + np.std(aligned_vals)**2) / 2)
            d = (np.mean(aligned_vals) - np.mean(base_vals)) / pooled_std if pooled_std > 0 else 0

            test_results.append({
                'family': family,
                'feature': feat,
                'base_mean': np.mean(base_vals),
                'aligned_mean': np.mean(aligned_vals),
                'cohens_d': d,
                'mann_whitney_U': stat,
                'p_value': p,
                'significant': p < 0.05 / (len(feature_cols) * len(families))  # Bonferroni
            })

            sig_marker = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {feat:25s}: base={np.mean(base_vals):.4f}, aligned={np.mean(aligned_vals):.4f}, "
                  f"d={d:.3f}, p={p:.2e} {sig_marker}")

    # Also test: base vs human, aligned vs human
    print("\n" + "=" * 70)
    print("BASE vs HUMAN and ALIGNED vs HUMAN (aggregated)")
    print("=" * 70)

    human_df = df[df['alignment'] == 'human']
    base_all_df = df[df['alignment'] == 'base']
    aligned_all_df = df[df['alignment'] == 'aligned']

    comparison_results = []
    for feat in feature_cols:
        h = human_df[feat].values
        b = base_all_df[feat].values
        a = aligned_all_df[feat].values

        stat_bh, p_bh = stats.mannwhitneyu(b, h, alternative='two-sided')
        stat_ah, p_ah = stats.mannwhitneyu(a, h, alternative='two-sided')

        pooled_bh = np.sqrt((np.std(b)**2 + np.std(h)**2) / 2)
        pooled_ah = np.sqrt((np.std(a)**2 + np.std(h)**2) / 2)
        d_bh = (np.mean(b) - np.mean(h)) / pooled_bh if pooled_bh > 0 else 0
        d_ah = (np.mean(a) - np.mean(h)) / pooled_ah if pooled_ah > 0 else 0

        comparison_results.append({
            'feature': feat,
            'human_mean': np.mean(h),
            'base_mean': np.mean(b),
            'aligned_mean': np.mean(a),
            'd_base_vs_human': d_bh,
            'd_aligned_vs_human': d_ah,
            'p_base_vs_human': p_bh,
            'p_aligned_vs_human': p_ah,
        })

        print(f"  {feat:25s}: human={np.mean(h):.4f}, base={np.mean(b):.4f} (d={d_bh:.3f}, p={p_bh:.2e}), "
              f"aligned={np.mean(a):.4f} (d={d_ah:.3f}, p={p_ah:.2e})")

    # Save results
    test_df = pd.DataFrame(test_results)
    test_df.to_csv(os.path.join(RESULTS_DIR, "exp1_base_vs_aligned_tests.csv"), index=False)
    comp_df = pd.DataFrame(comparison_results)
    comp_df.to_csv(os.path.join(RESULTS_DIR, "exp1_vs_human_tests.csv"), index=False)
    df.to_csv(os.path.join(RESULTS_DIR, "exp1_all_features.csv"), index=False)

    # ---- VISUALIZATIONS ----
    print("\nGenerating plots...")

    # Plot 1: Feature comparison across alignment types
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    for idx, feat in enumerate(feature_cols):
        ax = axes[idx // 3][idx % 3]
        order = ['human', 'base', 'aligned']
        palette = {'human': '#2ecc71', 'base': '#3498db', 'aligned': '#e74c3c'}
        sns.boxplot(data=df, x='alignment', y=feat, order=order, palette=palette, ax=ax)
        ax.set_title(feat.replace('_', ' ').title(), fontsize=11)
        ax.set_xlabel('')
    plt.suptitle('Distributional Features: Human vs Base vs Aligned', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "exp1_feature_boxplots.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Distinct n-gram ratios by model category
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, n in enumerate([1, 2, 3]):
        feat = f'distinct_{n}gram'
        ax = axes[idx]
        cat_order = ['human', 'base_mistral', 'aligned_mistral', 'base_mpt',
                     'aligned_mpt', 'base_cohere', 'aligned_cohere', 'chatgpt', 'gpt4']
        cat_colors = {
            'human': '#2ecc71', 'base_mistral': '#85c1e9', 'aligned_mistral': '#e74c3c',
            'base_mpt': '#85c1e9', 'aligned_mpt': '#e74c3c',
            'base_cohere': '#85c1e9', 'aligned_cohere': '#e74c3c',
            'chatgpt': '#c0392b', 'gpt4': '#8e44ad'
        }
        means = [df[df['category'] == c][feat].mean() for c in cat_order]
        stds = [df[df['category'] == c][feat].std() for c in cat_order]
        colors = [cat_colors[c] for c in cat_order]
        labels = [c.replace('_', '\n') for c in cat_order]

        bars = ax.bar(range(len(cat_order)), means, yerr=stds, color=colors, capsize=3, alpha=0.85)
        ax.set_xticks(range(len(cat_order)))
        ax.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
        ax.set_title(f'Distinct {n}-gram Ratio', fontsize=12)
        ax.set_ylabel('Ratio')
    plt.suptitle('Vocabulary Diversity: Base Models vs Aligned Models', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "exp1_ngram_diversity.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Effect sizes (Cohen's d) for base vs aligned per family
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = test_df.pivot(index='feature', columns='family', values='cohens_d')
    pivot.plot(kind='barh', ax=ax, color=['#e74c3c', '#3498db', '#2ecc71'])
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(x=-0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=-0.8, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0.8, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Cohen's d (aligned - base)")
    ax.set_title("Effect Sizes: Base vs Aligned Models by Feature", fontsize=13, fontweight='bold')
    ax.legend(title='Model Family')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "exp1_effect_sizes.png"), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nResults saved to {RESULTS_DIR}")
    print(f"Plots saved to {PLOTS_DIR}")
    print("Experiment 1 complete.")

    return df, test_df, comp_df


if __name__ == "__main__":
    run_experiment()
