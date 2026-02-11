# Research Plan: Alignment is the Watermark

## Motivation & Novelty Assessment

### Why This Research Matters
AI-generated text detection is increasingly critical for content integrity, education, and trust. Current detection methods are designed as external tools, but this research reframes the problem: alignment training itself—the process that makes LLMs helpful, harmless, and honest—inevitably creates a detectable distributional shift. If true, this means detection of aligned AI text is not merely a temporary arms race but a fundamental consequence of making AI useful. This insight has profound implications for AI policy, watermarking research, and the detection community.

### Gap in Existing Work
Based on the literature review, while individual papers have shown:
- RLHF increases detectability (Xu & Zubiaga, 2025)
- RLHF collapses diversity (Kirk et al., 2024)
- Reward models can serve as detectors (Lee et al., 2024)
- Detection is theoretically possible when TV > 0 (Chakraborty et al., 2023)

No paper has:
1. **Systematically tested the thesis that alignment = implicit watermark** across multiple model families using multiple detection methods
2. **Quantified the distributional gap** between base and aligned models using information-theoretic measures
3. **Used state-of-the-art LLMs as AI detectors** to test whether aligned models are more detectable by other AIs (as the hypothesis specifically states "at least by other AIs")
4. **Connected text-level statistical features** (vocabulary diversity, perplexity, burstiness) to detection performance in a unified framework

### Our Novel Contribution
We conduct a multi-method, multi-model empirical study that:
1. Compares base vs. aligned model texts on distributional features (linguistic fingerprints)
2. Tests multiple detection approaches on base vs. aligned model pairs from the RAID benchmark
3. Uses a state-of-the-art LLM (GPT-4.1) as an AI detector to test the "detectable by other AIs" claim
4. Quantifies the alignment watermark strength across model families

### Experiment Justification
- **Experiment 1 (Distributional Analysis):** Measures HOW alignment changes text distributions — vocabulary diversity, sentence structure, perplexity. Establishes that alignment creates a measurable distributional shift.
- **Experiment 2 (Statistical Detection):** Tests WHETHER this distributional shift translates to actual detectability differences. Uses zero-shot statistical detectors (perplexity, entropy, log-rank) on base vs. aligned pairs.
- **Experiment 3 (AI-as-Detector):** Tests the specific claim that aligned models are "detectable by other AIs." Uses GPT-4.1 as a classifier on base vs. aligned vs. human texts.
- **Experiment 4 (Cross-Family Generalization):** Tests whether the alignment watermark generalizes across model families (Mistral, MPT, Cohere) or is family-specific.

---

## Research Question
Does alignment training (RLHF, instruction tuning) create an inherent, detectable distributional shift—an implicit watermark—that makes aligned LLM outputs more AI-detectable than base model outputs, and is this detectable by other AIs?

## Hypothesis Decomposition

**H1 (Distributional Shift):** Aligned models produce text with lower vocabulary diversity, lower perplexity variance, and more formulaic structure compared to their base counterparts.

**H2 (Increased Detectability):** Aligned model outputs are significantly more detectable than base model outputs by zero-shot statistical detection methods.

**H3 (AI Detection):** A state-of-the-art LLM can distinguish aligned model text from human text with significantly higher accuracy than it can distinguish base model text from human text.

**H4 (Cross-Family Generality):** The alignment watermark phenomenon is consistent across different model families (not specific to one architecture).

## Proposed Methodology

### Approach
We use the RAID benchmark's base/chat model pairs (Mistral, MPT, Cohere) which provide a controlled comparison: same architecture, same prompts, different training. We apply three complementary detection paradigms:

1. **Linguistic feature analysis** — measure textual properties that differ between base/aligned/human text
2. **Statistical detection metrics** — apply established zero-shot detection signals (perplexity, entropy, log-rank)
3. **LLM-as-judge detection** — use GPT-4.1 to classify texts as human vs. AI

### Experimental Steps

1. **Data Preparation**
   - Load RAID base/chat pairs (Mistral, MPT, Cohere) — 500 samples each
   - Load human baseline (500 samples)
   - Compute text statistics (length, tokens, etc.)

2. **Experiment 1: Distributional Feature Analysis**
   - Compute per-text features: distinct n-gram ratios (1,2,3-gram), vocabulary richness (type-token ratio), mean word length, sentence length distribution, perplexity (using a small reference model)
   - Compare distributions across: human, base, aligned
   - Statistical tests: Mann-Whitney U for each feature between base and aligned

3. **Experiment 2: Zero-Shot Statistical Detection**
   - Use a local reference model (e.g., GPT-2 or small Llama) to compute per-token log-probabilities for each text
   - Compute detection signals: mean log-prob, log-rank, entropy
   - Evaluate AUROC for distinguishing human vs. base and human vs. aligned
   - Compare: is AUROC(human vs aligned) > AUROC(human vs base)?

4. **Experiment 3: LLM-as-Detector**
   - Sample 100 texts from each category (human, base-Mistral, chat-Mistral, base-MPT, chat-MPT, base-Cohere, chat-Cohere)
   - Prompt GPT-4.1 to classify each text as "human" or "AI-generated" with confidence
   - Compute accuracy, AUROC for each source category
   - Test: is detection accuracy higher for aligned vs. base?

5. **Experiment 4: Cross-Family Analysis**
   - Aggregate results from Experiments 1-3 across all three model families
   - Test consistency with paired comparisons
   - Identify family-specific vs. universal alignment signatures

### Baselines
- **Human text** (ground truth, should be least detectable)
- **Base model text** (no alignment, our "control")
- **Random baseline** (AUROC = 0.5, accuracy = 50%)

### Evaluation Metrics
- **AUROC** — primary metric, threshold-independent discrimination
- **Accuracy at 50% threshold** — practical binary classification
- **Cohen's d** — effect size for feature differences
- **Distinct n-gram ratios** — diversity measure (from Kirk et al.)
- **Type-token ratio** — vocabulary richness

### Statistical Analysis Plan
- **Mann-Whitney U test** for comparing feature distributions (non-parametric, handles non-normal data)
- **Bootstrap 95% confidence intervals** for AUROC estimates
- **Bonferroni correction** for multiple comparisons
- **Alpha = 0.05** significance level
- **Effect sizes** reported for all comparisons

## Expected Outcomes
- **H1 supported:** Aligned models show 30-60% lower vocabulary diversity than base models (per Kirk et al. findings)
- **H2 supported:** AUROC for detecting aligned models is 10-20 points higher than for base models (per Xu & Zubiaga)
- **H3 supported:** GPT-4.1 achieves >80% accuracy on aligned text vs. ~60% on base text
- **H4 supported:** Pattern is consistent across Mistral, MPT, and Cohere families

## Timeline and Milestones
1. Environment setup & data loading: 15 min
2. Experiment 1 (distributional analysis): 30 min
3. Experiment 2 (statistical detection with local model): 45 min
4. Experiment 3 (LLM-as-detector): 30 min
5. Experiment 4 (cross-family analysis): 15 min
6. Analysis & visualization: 30 min
7. Documentation: 30 min

## Potential Challenges
- **GPU memory:** We have 4x A6000s — ample for running local models
- **API costs:** Experiment 3 uses ~700 API calls to GPT-4.1; estimated $5-15
- **Base model quality:** RAID base model texts may be low quality, making the comparison partially about quality vs. alignment specifically

## Success Criteria
1. Clear, statistically significant difference in distributional features between base and aligned texts
2. Higher AUROC for detecting aligned vs. base model text across multiple detection methods
3. GPT-4.1 demonstrates higher classification accuracy on aligned text than base text
4. Results replicate across at least 2 of 3 model families
