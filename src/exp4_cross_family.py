"""
Experiment 4: Cross-Family Generalization Analysis
Aggregates results from Experiments 1-3 to test whether the alignment watermark
is consistent across model families (Mistral, MPT, Cohere).
"""
import json
import os
import sys
import numpy as np
import pandas as pd
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

FAMILIES = ['mistral', 'mpt', 'cohere']


def load_experiment_results():
    """Load results from all previous experiments."""
    results = {}

    # Exp 1: Distributional features
    path = os.path.join(RESULTS_DIR, "exp1_all_features.csv")
    if os.path.exists(path):
        results['exp1_features'] = pd.read_csv(path)

    path = os.path.join(RESULTS_DIR, "exp1_base_vs_aligned_tests.csv")
    if os.path.exists(path):
        results['exp1_tests'] = pd.read_csv(path)

    # Exp 2: Statistical detection
    path = os.path.join(RESULTS_DIR, "exp2_aggregate_auroc.csv")
    if os.path.exists(path):
        results['exp2_auroc'] = pd.read_csv(path)

    path = os.path.join(RESULTS_DIR, "exp2_detection_features.csv")
    if os.path.exists(path):
        results['exp2_features'] = pd.read_csv(path)

    # Exp 3: LLM detection
    path = os.path.join(RESULTS_DIR, "exp3_base_vs_aligned.csv")
    if os.path.exists(path):
        results['exp3_comparison'] = pd.read_csv(path)

    path = os.path.join(RESULTS_DIR, "exp3_accuracy_by_category.csv")
    if os.path.exists(path):
        results['exp3_accuracy'] = pd.read_csv(path)

    return results


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 4: Cross-Family Generalization Analysis")
    print("=" * 70)

    results = load_experiment_results()

    # ---- CONSISTENCY ANALYSIS ----
    print("\n" + "=" * 70)
    print("1. DIRECTION CONSISTENCY: Does alignment increase detectability in ALL families?")
    print("=" * 70)

    consistency_summary = []

    # Exp 1: Distributional features
    if 'exp1_tests' in results:
        exp1 = results['exp1_tests']
        key_features = ['type_token_ratio', 'distinct_2gram', 'distinct_3gram']

        for feat in key_features:
            feat_data = exp1[exp1['feature'] == feat]
            directions = []
            for _, row in feat_data.iterrows():
                # Negative d means aligned has lower diversity
                directions.append(row['cohens_d'])

            all_same_direction = all(d < 0 for d in directions) or all(d > 0 for d in directions)
            consistency_summary.append({
                'experiment': 'Exp1: Distributional',
                'measure': feat,
                'effect_sizes': directions,
                'consistent': all_same_direction,
                'families_agreeing': sum(1 for d in directions if d < 0) if sum(1 for d in directions if d < 0) >= 2 else sum(1 for d in directions if d > 0),
            })
            print(f"  {feat}: effects = {[f'{d:.3f}' for d in directions]}, consistent = {all_same_direction}")

    # Exp 2: Statistical detection AUROC
    if 'exp2_auroc' in results:
        exp2 = results['exp2_auroc']
        metrics = exp2['metric'].unique()

        for metric in metrics:
            metric_data = exp2[exp2['metric'] == metric]
            deltas = []
            for _, row in metric_data.iterrows():
                deltas.append(row['auroc_aligned'] - row['auroc_base'])

            all_positive = all(d > 0 for d in deltas)
            consistency_summary.append({
                'experiment': 'Exp2: Statistical Detection',
                'measure': f'AUROC({metric})',
                'effect_sizes': deltas,
                'consistent': all_positive,
                'families_agreeing': sum(1 for d in deltas if d > 0),
            })
            print(f"  AUROC({metric}): deltas = {[f'{d:+.4f}' for d in deltas]}, consistent = {all_positive}")

    # Exp 3: LLM detection
    if 'exp3_comparison' in results:
        exp3 = results['exp3_comparison']
        deltas = exp3['delta_tpr'].values.tolist()
        all_positive = all(d > 0 for d in deltas)
        consistency_summary.append({
            'experiment': 'Exp3: LLM Detection',
            'measure': 'TPR delta',
            'effect_sizes': deltas,
            'consistent': all_positive,
            'families_agreeing': sum(1 for d in deltas if d > 0),
        })
        print(f"  TPR deltas: {[f'{d:+.3f}' for d in deltas]}, consistent = {all_positive}")

    cons_df = pd.DataFrame(consistency_summary)
    cons_df.to_csv(os.path.join(RESULTS_DIR, "exp4_consistency.csv"), index=False)

    # ---- FAMILY-SPECIFIC ANALYSIS ----
    print("\n" + "=" * 70)
    print("2. FAMILY-SPECIFIC WATERMARK STRENGTH")
    print("=" * 70)

    family_strength = {f: [] for f in FAMILIES}

    # Aggregate effect sizes across experiments
    if 'exp1_tests' in results:
        exp1 = results['exp1_tests']
        for family in FAMILIES:
            fam_data = exp1[exp1['family'] == family]
            mean_abs_d = fam_data['cohens_d'].abs().mean()
            family_strength[family].append(('Exp1: mean |d|', mean_abs_d))

    if 'exp2_auroc' in results:
        exp2 = results['exp2_auroc']
        for family in FAMILIES:
            fam_data = exp2[exp2['family'] == family]
            mean_delta = fam_data['delta'].mean() if 'delta' in fam_data.columns else 0
            family_strength[family].append(('Exp2: mean AUROC delta', mean_delta))

    if 'exp3_comparison' in results:
        exp3 = results['exp3_comparison']
        for family in FAMILIES:
            fam_data = exp3[exp3['family'] == family]
            if len(fam_data) > 0:
                delta_tpr = fam_data['delta_tpr'].values[0]
                family_strength[family].append(('Exp3: TPR delta', delta_tpr))

    for family in FAMILIES:
        print(f"\n  {family.upper()}:")
        for measure, value in family_strength[family]:
            print(f"    {measure}: {value:+.4f}")

    # ---- SIGN TEST ----
    print("\n" + "=" * 70)
    print("3. SIGN TEST: Is the alignment watermark consistently positive?")
    print("=" * 70)

    # Collect all base-vs-aligned comparisons where positive = aligned more detectable
    all_deltas = []

    if 'exp2_auroc' in results:
        for _, row in results['exp2_auroc'].iterrows():
            all_deltas.append(row['auroc_aligned'] - row['auroc_base'])

    if 'exp3_comparison' in results:
        for _, row in results['exp3_comparison'].iterrows():
            all_deltas.append(row['delta_tpr'])

    if all_deltas:
        n_positive = sum(1 for d in all_deltas if d > 0)
        n_total = len(all_deltas)
        # Binomial test: H0: p(positive) = 0.5
        p_sign = stats.binomtest(n_positive, n_total, 0.5).pvalue

        print(f"  Total comparisons: {n_total}")
        print(f"  Positive (aligned more detectable): {n_positive}")
        print(f"  Binomial test p-value: {p_sign:.6f}")
        print(f"  Conclusion: {'Significant' if p_sign < 0.05 else 'Not significant'} "
              f"(aligned consistently more detectable)")

    # ---- COMPOSITE VISUALIZATION ----
    print("\nGenerating cross-family visualizations...")

    # Plot 1: Multi-panel summary
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: Exp1 effect sizes
    if 'exp1_tests' in results:
        ax = axes[0]
        exp1 = results['exp1_tests']
        key_feats = ['type_token_ratio', 'distinct_1gram', 'distinct_2gram', 'distinct_3gram',
                     'mean_sent_length', 'hapax_ratio']
        exp1_sub = exp1[exp1['feature'].isin(key_feats)]
        pivot = exp1_sub.pivot(index='feature', columns='family', values='cohens_d')
        pivot = pivot.reindex(columns=FAMILIES)
        pivot.plot(kind='barh', ax=ax, color=['#e74c3c', '#3498db', '#2ecc71'])
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel("Cohen's d (aligned - base)")
        ax.set_title("A) Distributional Features\n(Exp 1)", fontsize=12, fontweight='bold')
        ax.legend(title='Family')

    # Panel B: Exp2 AUROC
    if 'exp2_auroc' in results:
        ax = axes[1]
        exp2 = results['exp2_auroc']
        x = np.arange(len(FAMILIES))
        width = 0.25
        metrics = ['mean_logprob', 'mean_log_rank', 'mean_entropy']
        colors = ['#e74c3c', '#3498db', '#2ecc71']

        for i, metric in enumerate(metrics):
            metric_data = exp2[exp2['metric'] == metric]
            deltas = [metric_data[metric_data['family'] == f]['delta'].values[0]
                      if len(metric_data[metric_data['family'] == f]) > 0 else 0
                      for f in FAMILIES]
            ax.bar(x + i * width, deltas, width, label=metric.replace('_', ' '), color=colors[i], alpha=0.85)

        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_xticks(x + width)
        ax.set_xticklabels([f.capitalize() for f in FAMILIES])
        ax.set_ylabel('AUROC Delta (Aligned - Base)')
        ax.set_title("B) Detection AUROC Improvement\n(Exp 2)", fontsize=12, fontweight='bold')
        ax.legend(title='Metric')

    # Panel C: Exp3 TPR
    if 'exp3_comparison' in results:
        ax = axes[2]
        exp3 = results['exp3_comparison']
        x = np.arange(len(FAMILIES))
        width = 0.35

        base_vals = [exp3[exp3['family'] == f]['base_tpr'].values[0] for f in FAMILIES]
        aligned_vals = [exp3[exp3['family'] == f]['aligned_tpr'].values[0] for f in FAMILIES]

        ax.bar(x - width/2, base_vals, width, label='Base', color='#3498db', alpha=0.85)
        ax.bar(x + width/2, aligned_vals, width, label='Aligned', color='#e74c3c', alpha=0.85)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels([f.capitalize() for f in FAMILIES])
        ax.set_ylabel('Detection Rate (TPR)')
        ax.set_title("C) GPT-4.1 Detection\n(Exp 3)", fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.legend()

    plt.suptitle('Cross-Family Consistency: Alignment as Implicit Watermark',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "exp4_cross_family_summary.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Heatmap of effect directions
    if 'exp1_tests' in results and 'exp2_auroc' in results:
        fig, ax = plt.subplots(figsize=(8, 8))

        measures = []
        data_matrix = []

        if 'exp1_tests' in results:
            exp1 = results['exp1_tests']
            for feat in exp1['feature'].unique():
                row = []
                for family in FAMILIES:
                    val = exp1[(exp1['feature'] == feat) & (exp1['family'] == family)]['cohens_d'].values
                    row.append(val[0] if len(val) > 0 else 0)
                measures.append(f'Exp1: {feat}')
                data_matrix.append(row)

        if 'exp2_auroc' in results:
            for metric in results['exp2_auroc']['metric'].unique():
                row = []
                for family in FAMILIES:
                    val = results['exp2_auroc'][
                        (results['exp2_auroc']['metric'] == metric) &
                        (results['exp2_auroc']['family'] == family)
                    ]['delta'].values if 'delta' in results['exp2_auroc'].columns else [0]
                    row.append(val[0] if len(val) > 0 else 0)
                measures.append(f'Exp2: {metric}')
                data_matrix.append(row)

        if 'exp3_comparison' in results:
            row = []
            for family in FAMILIES:
                val = results['exp3_comparison'][results['exp3_comparison']['family'] == family]['delta_tpr'].values
                row.append(val[0] if len(val) > 0 else 0)
            measures.append('Exp3: GPT-4.1 TPR')
            data_matrix.append(row)

        matrix = np.array(data_matrix)
        sns.heatmap(matrix, xticklabels=[f.capitalize() for f in FAMILIES],
                    yticklabels=measures, annot=True, fmt='.3f', cmap='RdBu_r',
                    center=0, ax=ax)
        ax.set_title('Effect Direction Heatmap\n(Positive = Aligned More Detectable)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "exp4_effect_heatmap.png"), dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\nResults saved to {RESULTS_DIR}")
    print(f"Plots saved to {PLOTS_DIR}")
    print("Experiment 4 complete.")

    return cons_df


if __name__ == "__main__":
    run_experiment()
