# Alignment is the Watermark

**Empirical evidence that alignment training creates an inherent, detectable signature in LLM outputs.**

## Key Findings

- **Alignment dramatically increases detectability:** Zero-shot detection AUROC increases by +0.09 to +0.35 when comparing aligned vs. base model text across three model families (Mistral, MPT, Cohere)
- **Base models are near-undetectable:** Base Mistral and MPT achieve AUROC ~0.50 (random chance) on log-probability-based detection, while their aligned variants reach 0.71-0.88
- **GPT-4.1 detects aligned text near-perfectly:** 97.5-100% true positive rate on aligned models vs. 85-96.3% on base models
- **Cross-family consistency:** 11/12 detection comparisons show aligned > base (sign test p=0.006)
- **The watermark is structural:** Alignment consistently reduces sentence length variability (Cohen's d = -0.45 to -0.66), creating predictable, well-structured prose

## Project Structure

```
alignment-watermark-0988-claude/
├── REPORT.md              # Full research report with results
├── README.md              # This file
├── planning.md            # Research plan and methodology
├── literature_review.md   # Comprehensive literature synthesis (26 papers)
├── resources.md           # Resource catalog
├── src/                   # Experiment implementations
│   ├── exp1_distributional_analysis.py   # Linguistic feature analysis
│   ├── exp2_statistical_detection.py     # Zero-shot detection (GPT-2 reference model)
│   ├── exp3_llm_detector.py             # GPT-4.1 as AI detector
│   └── exp4_cross_family.py             # Cross-family generalization
├── results/
│   ├── data/              # CSV results and raw classifications
│   └── plots/             # All visualizations (13 plots)
├── datasets/raid/         # RAID benchmark data (3 base/chat pairs + human)
├── papers/                # 26 downloaded research papers
└── code/                  # 6 cloned reference repositories
```

## Reproduce

```bash
# Setup
source .venv/bin/activate

# Run all experiments
python src/exp1_distributional_analysis.py  # ~30 seconds, CPU only
CUDA_VISIBLE_DEVICES=0 LOGNAME=researcher USER=researcher python src/exp2_statistical_detection.py  # ~4 min, needs GPU
python src/exp3_llm_detector.py             # ~8 min, needs OPENAI_API_KEY
python src/exp4_cross_family.py             # ~5 seconds, aggregation only
```

**Requirements:** Python 3.12+, NVIDIA GPU (48GB recommended), OpenAI API key for Experiment 3.

See [REPORT.md](REPORT.md) for full methodology, results, and analysis.
