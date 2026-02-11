# Resources Catalog: Alignment is the Watermark

## Overview

This document catalogs all resources gathered for the research project "Alignment is the Watermark." Resources include academic papers, datasets, and code repositories organized by relevance to the central hypothesis.

---

## Papers (26 total)

All papers are stored in `papers/` as PDF files.

### Tier 1: Directly Relevant to Hypothesis

| File | Title | Authors | Year | Venue | Relevance |
|------|-------|---------|------|-------|-----------|
| `xu2025_rlhf_detectability.pdf` | Understanding the Effects of RLHF on the Quality and Detectability of LLM-Generated Texts | Xu & Zubiaga | 2025 | arXiv:2503.17965 | **Most directly relevant.** Tests detectability across base/SFT/PPO stages. Shows AUROC increases from 0.68 to 0.91 with alignment. |
| `lee2024_remodetect.pdf` | ReMoDetect: Reward Models Recognize Aligned LLM's Generations | Lee, Tack & Shin | 2024 | -- | Reward models as detectors; aligned models 95-99% AUROC vs. 60-75% for base. |
| `kirk2023_rlhf_diversity.pdf` | Understanding the Effects of RLHF on LLM Generalisation and Diversity | Kirk et al. | 2024 | ICLR 2024 | RLHF reduces diversity 75-90%; first demonstration of across-input mode collapse. |
| `rivera2025_distinct_style.pdf` | AI-generated Text has a Distinct Style | Rivera-Soto et al. | 2025 | -- | Measurable stylistic fingerprints in AI text. |
| `fraser2024_factors_detectability.pdf` | Factors Affecting the Detectability of AI-Generated Text | Fraser et al. | 2024 | -- | Systematic analysis of detectability factors including alignment. |
| `mcgovern2024_fingerprints.pdf` | Fingerprints of AI-Generated Text | McGovern et al. | 2024 | -- | Persistent linguistic fingerprints across models. |
| `panickssery2024_self_recognition.pdf` | LLM Evaluators Recognize and Favor Their Own Generations | Panickssery et al. | 2024 | -- | Self-recognition implies model-specific alignment signatures. |

### Tier 2: Detection Methods

| File | Title | Authors | Year | Venue | Key Contribution |
|------|-------|---------|------|-------|-----------------|
| `2301.11305_detectgpt.pdf` | DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature | Mitchell et al. | 2023 | ICML 2023 | Probability curvature detection; "LLMs watermark themselves implicitly." |
| `2310.05130_fast_detectgpt.pdf` | Fast-DetectGPT: Efficient Zero-Shot Detection via Conditional Probability Curvature | Bao et al. | 2024 | ICLR 2024 | 340x faster curvature detection; primary detector in Xu & Zubiaga. |
| `1906.04043_gltr.pdf` | GLTR: Statistical Detection and Visualization of Generated Text | Gehrmann et al. | 2019 | ACL 2019 | Statistical token analysis. |
| `2305.17359_dna_gpt.pdf` | DNA-GPT: Divergent N-Gram Analysis for Training-Free Detection | Yang et al. | 2023 | arXiv | Divergence-based detection. |
| `2306.04723_intrinsic_dimension.pdf` | Intrinsic Dimension Estimation for Robust Detection of AI-Generated Texts | Tulchinskii et al. | 2023 | arXiv | Intrinsic dimensionality as detection signal. |
| `2309.13322_text_to_source.pdf` | From Text to Source: Results in Detecting LLM-Generated Content | -- | 2023 | arXiv | Source attribution methods. |
| `bhattacharjee2023_chatgpt_detector.pdf` | Fighting Fire with Fire: Can ChatGPT Detect Its Own Text? | Bhattacharjee et al. | 2023 | -- | LLM self-detection capabilities. |

### Tier 3: Detection Impossibility and Possibility

| File | Title | Authors | Year | Venue | Key Contribution |
|------|-------|---------|------|-------|-----------------|
| `2304.04736_possibilities_detection.pdf` | On the Possibilities of AI-Generated Text Detection | Chakraborty et al. | 2023 | arXiv | Theoretical framework: detection possible iff TV(m,h)>0; multi-sample scaling. |
| `2303.11156_reliable_detection.pdf` | Can AI-Generated Text be Reliably Detected? | Sadasivan et al. | 2023 | arXiv | Single-sample impossibility argument. |
| `1911.00650_detection_easiest.pdf` | Automatic Detection is Easiest When Humans are Fooled | Ippolito et al. | 2020 | ACL 2020 | Detection vs. human perception trade-off. |

### Tier 4: Benchmarks and Surveys

| File | Title | Authors | Year | Venue | Key Contribution |
|------|-------|---------|------|-------|-----------------|
| `2405.07940_raid.pdf` | RAID: Robust Evaluation of Machine-Generated Text Detectors | Dugan et al. | 2024 | ACL 2024 | Largest benchmark; 11 LLMs with base/chat pairs. |
| `2303.14822_mgtbench.pdf` | MGTBench: Benchmarking Machine-Generated Text Detection | He et al. | 2024 | CCS 2024 | 13 detection methods comparison. |
| `2305.13242_mage.pdf` | MAGE: Machine-Generated Text Detection in the Wild | Li et al. | 2024 | ACL 2024 | 27 LLMs, 10 domains, 8 testbeds. |
| `2301.07597_hc3.pdf` | How Close is ChatGPT to Human Experts? | Guo et al. | 2023 | arXiv | Human ChatGPT Comparison Corpus. |

### Tier 5: Watermarking and Alignment Methods

| File | Title | Authors | Year | Venue | Key Contribution |
|------|-------|---------|------|-------|-----------------|
| `2301.10226_watermark_llm.pdf` | A Watermark for Large Language Models | Kirchenbauer et al. | 2023 | ICML 2023 | Green/red list watermarking. |
| `2306.04634_watermark_reliability.pdf` | On the Reliability of Watermarks for LLMs | Kirchenbauer et al. | 2024 | ICLR 2024 | Watermark robustness analysis. |
| `dathathri2024_synthid.pdf` | SynthID: Scalable Watermarking for LLM Outputs | Dathathri et al. | 2024 | Nature | Google's production watermarking. |
| `2203.02155_instructgpt.pdf` | Training Language Models to Follow Instructions with Human Feedback | Ouyang et al. | 2022 | NeurIPS 2022 | Foundational RLHF paper. |
| `li2024_predicting_vs_acting.pdf` | Predicting vs. Acting | Li et al. | 2024 | -- | Base model knowledge vs. alignment generation patterns. |

---

## Datasets

### Primary: RAID Dataset
**Location:** `datasets/raid/`

The RAID dataset is the primary resource for our experiments due to its explicit base vs. chat model pairs.

| File | Description | Size |
|------|-------------|------|
| `raid_clean_samples.json` | 6,000 clean samples (500 per model, attack='none'), 12 models | 11 MB |
| `raid_extra_samples.json` | 2,100 extra samples from extra split | 3.8 MB |
| `pair_mistral.json` | 500 Mistral base + 500 Mistral Chat samples | 1.9 MB |
| `pair_mpt.json` | 500 MPT base + 500 MPT Chat samples | 2.0 MB |
| `pair_cohere.json` | 500 Cohere base + 500 Cohere Chat samples | 1.6 MB |
| `dataset_summary.json` | Summary statistics for all 12 models | -- |
| `README.md` | Comprehensive documentation | -- |

**Models in RAID (12 total):**
- Base: GPT-2 XL, GPT-3, Mistral-7B, MPT-30B, Cohere
- Chat/Aligned: ChatGPT, GPT-4, Mistral-7B Chat, MPT-30B Chat, Cohere Chat, LLaMA 2 70B Chat
- Human-written baseline

**Base vs. Chat Pairs for Direct Comparison:**
1. Mistral-7B / Mistral-7B Chat
2. MPT-30B / MPT-30B Chat
3. Cohere / Cohere Chat

**Full RAID dataset available via HuggingFace:**
```python
from datasets import load_dataset
ds = load_dataset("liamdugan/raid", split="train")
```

### Recommended Additional Datasets (not yet downloaded)

| Dataset | Source | Why Relevant |
|---------|--------|-------------|
| MAGE | `datasets.load_dataset("yaful/MAGE")` | 27 LLMs, 10 domains |
| HC3 | HuggingFace | Parallel human/ChatGPT answers |
| Anthropic HH-RLHF | HuggingFace | Human preference data for reward model training |

---

## Code Repositories

All repositories are cloned into `code/`.

### 1. RAID Benchmark
**Path:** `code/raid/`
**GitHub:** https://github.com/liamdugan/raid
**Paper:** ACL 2024
**Description:** Evaluation framework for AI text detectors. Includes CLI tools for detection and evaluation, detector implementations, and leaderboard submission scripts.
**Key files:** `detect_cli.py`, `evaluate_cli.py`, `raid/detect.py`, `raid/evaluate.py`
**Dependencies:** pandas, numpy, torch, transformers, scikit-learn, datasets, lightgbm

### 2. DetectGPT
**Path:** `code/detect-gpt/`
**GitHub:** https://github.com/eric-mitchell/detect-gpt
**Paper:** ICML 2023
**Description:** Zero-shot detection via probability curvature using perturbation discrepancy.
**Key files:** `run.py`, `custom_datasets.py`, `paper_scripts/main.sh`
**Dependencies:** torch, numpy, transformers, datasets, openai

### 3. Fast-DetectGPT
**Path:** `code/fast-detect-gpt/`
**GitHub:** https://github.com/baoguangsheng/fast-detect-gpt
**Paper:** ICLR 2024
**Description:** Efficient zero-shot detection via conditional probability curvature. 340x speedup over DetectGPT.
**Key files:** `scripts/fast_detect_gpt.py`, `scripts/detect_gpt.py`, `scripts/local_infer.py`
**Dependencies:** torch, numpy, transformers (4.28.1), datasets (2.12.0), nltk
**Notes:** Requires Tesla A100 GPU (80G). Includes pre-generated data for GPT-3/ChatGPT/GPT-4.

### 4. MGTBench
**Path:** `code/MGTBench/`
**GitHub:** https://github.com/xinleihe/MGTBench
**Paper:** ACM CCS 2024
**Description:** Benchmark with 13 detection methods. Supports metric-based (Log-Likelihood, Rank, LogRank, Entropy, GLTR, DetectGPT, LRR, NPR) and model-based (OpenAI Detector, ChatGPT Detector, ConDA, GPTZero, LM Detector) methods.
**Key files:** `benchmark.py`, `methods/metric_based.py`, `methods/detectgpt.py`
**Dependencies:** PyTorch 1.13.1, TensorFlow 2.10, transformers 4.24.0, textattack (heavy conda env)

### 5. MAGE
**Path:** `code/MAGE/`
**GitHub:** https://github.com/yafuly/MAGE
**Paper:** ACL 2024
**Description:** Multi-generator detection with 27 LLMs, 10 domains. Provides systematic testbeds.
**Key files:** `training/longformer/`, `deployment/prepare_testbeds.py`
**Dependencies:** accelerate, transformers (4.35.2), torch, clean-text, ftfy

### 6. LM-Watermarking
**Path:** `code/lm-watermarking/`
**GitHub:** https://github.com/jwkirchenbauer/lm-watermarking
**Paper:** ICML 2023 / ICLR 2024
**Description:** Explicit watermarking via green/red list token partitioning. Useful as comparison baseline for implicit alignment watermark.
**Key files:** `watermark_processor.py`, `extended_watermark_processor.py`, `demo_watermark.py`
**Dependencies:** gradio, nltk, scipy, torch, transformers

### Common Dependencies Across Repos
- PyTorch (1.13.1+)
- HuggingFace Transformers (4.24+)
- NumPy
- datasets (HuggingFace)
- Most require GPU (A100 recommended)

---

## Paper Search Results

Raw search results from paper-finder are stored in `paper_search_results/` as JSONL files:

| File | Query |
|------|-------|
| `AI_text_detection_aligned_language_models_vs_base_models_*.jsonl` | Base vs. aligned model detection |
| `RLHF_reinforcement_learning_human_feedback_text_detectability_*.jsonl` | RLHF and detectability |
| `machine_generated_text_detection_survey_benchmark_*.jsonl` | Detection surveys and benchmarks |
| `base_model_vs_instruction_tuned_model_distribution_shift_*.jsonl` | Distribution shift from alignment |

---

## Quick Start Guide

### Running Detection Experiments

1. **RAID evaluation (recommended starting point):**
```bash
cd code/raid
pip install -r requirements.txt
python detect_cli.py --detector fast-detectgpt --data ../datasets/raid/pair_mistral.json
```

2. **Fast-DetectGPT on custom data:**
```bash
cd code/fast-detect-gpt
bash setup.sh
python scripts/fast_detect_gpt.py --scoring_model_name <model> --data_path <path>
```

3. **MGTBench multi-method comparison:**
```bash
cd code/MGTBench
python benchmark.py --dataset Essay --method Log-Likelihood
```

### Key Experiment: Base vs. Aligned Detection

Using the RAID pair files for direct comparison:
```python
import json

# Load base/chat pairs
with open("datasets/raid/pair_mistral.json") as f:
    pairs = json.load(f)

base_texts = [s for s in pairs if s["model"] == "mistral-7b"]
chat_texts = [s for s in pairs if s["model"] == "mistral-7b-chat"]

# Run detector on both sets and compare AUROC
```

---

## File Structure

```
alignment-watermark-0988-claude/
├── literature_review.md          # Comprehensive literature synthesis
├── resources.md                  # This file
├── papers/                       # 26 downloaded PDFs
│   ├── pages/                    # Chunked PDFs for reading
│   └── *.pdf
├── datasets/
│   └── raid/                     # RAID dataset samples and pairs
│       ├── raid_clean_samples.json
│       ├── raid_extra_samples.json
│       ├── pair_mistral.json
│       ├── pair_mpt.json
│       ├── pair_cohere.json
│       └── README.md
├── code/                         # 6 cloned repositories
│   ├── raid/
│   ├── detect-gpt/
│   ├── fast-detect-gpt/
│   ├── MGTBench/
│   ├── MAGE/
│   └── lm-watermarking/
├── paper_search_results/         # Raw search results
└── pyproject.toml                # Project configuration
```
