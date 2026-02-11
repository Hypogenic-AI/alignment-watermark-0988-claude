# Literature Review: Alignment is the Watermark

## Research Hypothesis

> Base language models trained solely to match data distributions are not AI detectable, as they replicate the distribution itself. In contrast, aligned models -- regardless of the specific definition of alignment -- are likely to be AI detectable due to their characteristic choices in language, honesty, and competence.

This review synthesizes findings from 26 papers spanning AI-generated text detection, alignment training (RLHF/DPO/SFT), text watermarking, and distributional analysis of LLM outputs. The literature is organized around the central question: **does alignment training create an inherent, detectable signature in LLM outputs?**

---

## 1. Core Evidence: Alignment Increases Detectability

### 1.1 Xu & Zubiaga (2025) -- Direct Evidence

**Paper:** "Understanding the Effects of RLHF on the Quality and Detectability of LLM-Generated Texts" (arXiv:2503.17965)

This is the most directly relevant paper, testing the exact hypothesis. Using Llama-7B at three alignment stages (base, SFT, PPO), the authors measure both output quality and detectability.

**Key results for instruction following (general text):**

| Detector | Llama (base) | Llama_SFT | Llama_PPO |
|----------|-------------|-----------|-----------|
| Fast-DetectGPT (AUROC) | 0.68 | 0.80 | **0.91** |
| GPTZero (AUROC) | 0.59 | 0.70 | **0.81** |

The monotonic increase from base to PPO -- a 23-point AUROC improvement for Fast-DetectGPT -- provides direct evidence that alignment progressively increases detectability.

**Diversity collapse as mechanism:**
- Distinct n-gram scores: base=0.42, SFT=0.24, PPO=**0.18** (57% reduction)
- RLHF produces "lengthier and more repetitive" outputs with "decreased syntactic and semantic diversity"
- The model "internalizes and reproduces features inherently associated with LLM generation"

**Quality-detectability trade-off:** PPO achieves 20.62% AlpacaEval win rate (vs. 1.14% for base) while simultaneously becoming far more detectable. Better alignment = better quality = more detectable.

**Caveat:** For question answering with mixed code/text (StackExchange), detectability *decreases* with alignment (Fast-DetectGPT AUROC: 0.91 base -> 0.84 PPO), suggesting the effect is strongest for natural language generation.

### 1.2 Kirk et al. (2024) -- RLHF Diversity Collapse

**Paper:** "Understanding the Effects of RLHF on LLM Generalisation and Diversity" (ICLR 2024, arXiv:2310.06452)

This paper provides the most rigorous empirical evidence that RLHF systematically narrows the output distribution. While not focused on detection, its findings have profound implications.

**Per-input diversity collapse (LLaMA 7B, summarization):**
- EAD (distinct n-grams): RLHF ~0.2 vs. SFT ~0.8 (**75% reduction**)
- SentenceBERT diversity: RLHF ~0.05 vs. SFT ~0.5 (**90% reduction**)

**Across-input mode collapse (first rigorous demonstration):**
- Even for different inputs, RLHF outputs converge to a consistent "mode"
- EAD: RLHF ~0.83 vs. SFT ~0.89
- The authors state this is "the first rigorous empirical demonstration of across-input mode collapse emerging from RLHF training"

**Critical findings:**
1. The KL penalty (standard RLHF countermeasure) **fails to recover diversity** -- increasing it actually reduces per-input diversity further
2. Best-of-N sampling preserves diversity while improving quality, localizing the collapse to PPO optimization specifically
3. Effects are consistent across model scales (LLaMA 7B and OPT at multiple sizes)

**Implication for detection:** Reduced per-input diversity means more predictable patterns. Across-input mode collapse creates a consistent stylistic fingerprint. Both are exploitable detection signals.

### 1.3 Lee, Tack & Shin (2024) -- ReMoDetect

**Paper:** "ReMoDetect: Reward Models Recognize Aligned LLM's Generations"

This paper demonstrates that reward models -- the very models trained during RLHF -- can serve as highly effective AI text detectors.

**Core insight:** Alignment training pushes LLM outputs into high-reward regions of text space. A reward model can detect this by simply thresholding on reward score: aligned outputs cluster at high values, human text does not.

**Detection performance:**
- On ChatGPT outputs: ~95-99% AUROC
- On LLaMA-2-Chat outputs: ~92-98% AUROC
- On base (unaligned) models: **~60-75% AUROC** (much weaker)

**Key implications:**
- Cross-model transfer works: a reward model from one family detects outputs from differently-aligned LLMs, suggesting alignment methods converge on similar distributional signatures
- Stronger alignment = stronger detection signal (monotonic relationship)
- The reward model serves as the natural "detection key" for the implicit alignment watermark
- Detection is robust to paraphrasing attacks

---

## 2. Theoretical Foundations

### 2.1 Chakraborty et al. (2023) -- Information-Theoretic Possibility

**Paper:** "On the Possibilities of AI-Generated Text Detection" (arXiv:2304.04736)

This paper provides the theoretical framework for when detection is possible vs. impossible.

**Central theorem:** Detection is possible if and only if TV(m, h) > 0 (the total variation distance between machine and human distributions is non-zero). The only case where detection is truly impossible is when the distributions are **identical** across their entire support.

**Multi-sample detection:** For any non-zero TV distance delta:
- Sample complexity: n = Omega(1/delta^2 * log(1/(1-epsilon)))
- Single-sample AUROC upper bound: 1/2 + TV - TV^2/2
- With n samples, TV(m^n, h^n) -> 1 exponentially

**Quantitative scaling (Figure 1, for TV=0.1):**
- n=1: AUROC ~0.6 (near random)
- n=30: AUROC ~0.7
- n=100: AUROC ~0.85
- n=500: AUROC ~0.98 (near perfect)

**Connection to alignment hypothesis:** The paper observes that "it would be quite difficult to make LLMs exactly equal to human distributions due to the vast diversity within the human population." Alignment makes this not just difficult but *intentionally impossible* -- alignment IS the mechanism ensuring the distributional gap persists. A hypothetical perfect base model (TV=0) would be undetectable but also unaligned; the moment you align it, you guarantee TV>0 and thus detectability.

### 2.2 Mitchell et al. (2023) -- DetectGPT

**Paper:** "DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature" (ICML 2023, arXiv:2301.11305)

DetectGPT introduces the perturbation discrepancy as a detection signal: machine-generated text sits in negative-curvature regions of the model's log probability function.

**The implicit watermark insight:** The paper explicitly states: "LLMs that do not perfectly imitate human writing essentially watermark themselves implicitly." This is the foundational observation for our hypothesis.

**Mathematical basis:**
- Perturbation discrepancy: d(x, p_theta, q) = log p_theta(x) - E[log p_theta(x_tilde)]
- Approximates the negative trace of the Hessian of log p_theta (curvature measure)
- Machine text sits near local maxima; perturbations move downhill
- Human text sits in flatter regions; perturbations move in mixed directions

**Results (AUROC, averaged across 5 base models):**

| Method | XSum | SQuAD | WritingPrompts |
|--------|------|-------|----------------|
| Log p(x) | 0.83 | 0.82 | 0.95 |
| DetectGPT | **0.97** | **0.92** | **0.97** |

**Gap in the paper:** All models tested are base models only. The authors speculate that RLHF "may exacerbate" the disconnect between model and human distributions but do not test this. Our hypothesis predicts that aligned models would produce even sharper curvature signatures.

### 2.3 Bao et al. (2024) -- Fast-DetectGPT

**Paper:** "Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature" (ICLR 2024, arXiv:2310.05130)

Fast-DetectGPT replaces DetectGPT's perturbation-based curvature estimation with conditional probability curvature, achieving 340x speedup while improving accuracy. Uses sampling/scoring model pairs (e.g., Llama3-8B/Llama3-8B-Instruct).

This is the detector that shows the strongest alignment-dependent improvement in Xu & Zubiaga (2025): AUROC 0.68 -> 0.91 from base to PPO on instruction following.

---

## 3. Detection Benchmarks and Empirical Landscape

### 3.1 Dugan et al. (2024) -- RAID Benchmark

**Paper:** "RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors" (ACL 2024, arXiv:2405.07940)

RAID is the largest detection benchmark: 6.2M+ generations across 11 LLMs, 8 domains, 4 decoding strategies, and 11 adversarial attacks.

**Critical for our hypothesis -- base vs. chat model pairs:**

| Model Family | Base Variant | Chat Variant |
|-------------|-------------|--------------|
| Mistral-7B | Mistral-7B | Mistral-7B Chat |
| MPT-30B | MPT-30B | MPT-30B Chat |
| Cohere | Cohere (command) | Cohere Chat |

**Detection accuracy at FPR=5%:**
- Chat/aligned models are **consistently easier to detect** than base counterparts
- Base models with sampling + repetition penalty are hardest to detect (some detectors drop to ~0% accuracy)
- Repetition penalty decreases accuracy by up to 32 percentage points

**Self-BLEU statistics (lower = less repetitive):**
- Mistral base: 19.1 vs. Mistral Chat: 9.16
- MPT base: 22.1 vs. MPT Chat: 5.39

**Key finding:** "Even strong detectors can catastrophically fail" -- simply changing the generator, decoding strategy, or applying repetition penalty introduces up to 95%+ error rates.

### 3.2 He et al. (2023) -- MGTBench

**Paper:** "MGTBench: Benchmarking Machine-Generated Text Detection" (ACM CCS 2024, arXiv:2303.14822)

Comprehensive benchmark with 13 detection methods (metric-based and model-based) across multiple generators and datasets.

### 3.3 Li et al. (2024) -- MAGE

**Paper:** "MAGE: Machine-Generated Text Detection in the Wild" (ACL 2024, arXiv:2305.13242)

Largest-scale multi-generator detection study with 447,674 texts from 27 LLMs across 10 domains and 8 systematic testbeds of increasing difficulty.

### 3.4 Guo et al. (2023) -- HC3

**Paper:** "How Close is ChatGPT to Human Experts?" (arXiv:2301.07597)

Human ChatGPT Comparison Corpus with parallel human and ChatGPT answers, useful for direct base vs. aligned comparison studies.

---

## 4. Alignment Training Methods and Their Effects

### 4.1 Ouyang et al. (2022) -- InstructGPT

**Paper:** "Training Language Models to Follow Instructions with Human Feedback" (NeurIPS 2022, arXiv:2203.02155)

The foundational RLHF paper establishing the SFT -> Reward Modeling -> PPO pipeline. Demonstrates that alignment dramatically changes output characteristics: instruction-following, safety, format compliance. Each stage represents a measurable distributional shift from the base model.

### 4.2 Kirchenbauer et al. (2023a, 2023b) -- LLM Watermarking

**Papers:**
- "A Watermark for Large Language Models" (ICML 2023, arXiv:2301.10226)
- "On the Reliability of Watermarks for Large Language Models" (ICLR 2024, arXiv:2306.04634)

These papers on explicit watermarking provide an instructive contrast to our implicit watermark hypothesis:
- **Explicit watermarks** modify the generation process (green/red list token partitioning, logit bias delta=2.0)
- **Implicit alignment watermark** emerges naturally from the RLHF training objective
- Both create detectable distributional shifts; the key difference is intentionality
- Explicit watermarks are vulnerable to paraphrasing; the alignment watermark may be more robust because it is embedded in the model's learned preferences, not just the token selection mechanism

### 4.3 Dathathri et al. (2024) -- SynthID

**Paper:** "Scalable Watermarking for Identifying Large Language Model Outputs" (Nature 2024)

Google's production watermarking system using tournament sampling. Demonstrates that practical watermarking at scale is possible, but requires control over the generation process. Our hypothesis suggests aligned models already carry a natural version of this signal.

---

## 5. Additional Supporting Evidence

### 5.1 Rivera-Soto et al. (2025) -- Distinct Style

**Paper:** "AI-generated Text has a Distinct Style Which is Influenced by the Prompt"

Demonstrates that AI-generated text has measurable stylistic properties that differ from human text, and that these properties are influenced by but not fully determined by the prompt.

### 5.2 McGovern et al. (2024) -- Fingerprints

**Paper:** "Fingerprints of AI-Generated Text"

Identifies specific linguistic fingerprints in AI-generated text that persist across models and domains.

### 5.3 Fraser et al. (2024) -- Factors of Detectability

**Paper:** "Factors Affecting the Detectability of AI-Generated Text"

Systematic study of what makes some AI text more detectable than others. Factors include model size, decoding strategy, domain, and -- crucially -- alignment training.

### 5.4 Panickssery et al. (2024) -- Self-Recognition

**Paper:** "LLM Evaluators Recognize and Favor Their Own Generations"

LLMs can recognize their own outputs, suggesting alignment creates model-specific signatures that are inherently discriminable.

### 5.5 Li et al. (2024) -- Predicting vs. Acting

**Paper:** "Predicting vs. Acting: A Trade-off Between World Knowledge and Language Generation"

Explores the tension between a model's world knowledge (base model capability) and its generation patterns (alignment-influenced), relevant to understanding why alignment shifts distributions.

### 5.6 Bhattacharjee et al. (2023) -- ChatGPT Detector

**Paper:** "Fighting Fire with Fire: Can ChatGPT Be Used to Detect ChatGPT-Generated Text?"

Early work on using LLMs themselves as detectors, finding that ChatGPT can detect its own outputs with reasonable accuracy.

---

## 6. Detection Methods Taxonomy

### 6.1 Zero-Shot Statistical Methods
- **DetectGPT** (Mitchell et al., 2023): Probability curvature via perturbation
- **Fast-DetectGPT** (Bao et al., 2024): Conditional probability curvature (340x faster)
- **DNA-GPT** (Yang et al., 2023): Divergence-based detection
- **GLTR** (Gehrmann et al., 2019): Statistical token analysis and visualization
- **Binoculars** (Hans et al., 2024): Cross-model perplexity comparison

### 6.2 Trained Classifiers
- **RoBERTa-based detectors** (OpenAI): Fine-tuned on labeled human/machine data
- **RADAR** (Hu et al., 2023): Adversarially-trained robust detector
- **ReMoDetect** (Lee et al., 2024): Reward model as detector

### 6.3 Proactive Methods
- **Green/Red List Watermarking** (Kirchenbauer et al., 2023)
- **SynthID** (Dathathri et al., 2024): Tournament sampling watermark
- Our hypothesis: **Alignment itself** as a natural watermark

### 6.4 Benchmark Frameworks
- **RAID** (Dugan et al., 2024): 11 models, 8 domains, adversarial attacks
- **MGTBench** (He et al., 2023): 13 detection methods
- **MAGE** (Li et al., 2024): 27 generators, 10 domains

---

## 7. Key Methodological Insights for Experiments

### 7.1 Recommended Evaluation Metrics
- **AUROC** as primary metric (used across all surveyed papers)
- **Accuracy at FPR=5%** for practical deployment scenarios (RAID standard)
- **TPR at fixed FPR** (1%, 5%, 10%) for threshold sensitivity
- **Distinct n-gram scores** and **SentenceBERT similarity** for diversity measurement

### 7.2 Critical Experimental Controls
1. **Same model family, different alignment stages:** Llama base -> SFT -> PPO (Xu & Zubiaga design)
2. **Base vs. chat pairs:** Mistral/Mistral-Chat, MPT/MPT-Chat, Cohere/Cohere-Chat (RAID design)
3. **Multiple detectors:** At minimum Fast-DetectGPT (zero-shot) + RoBERTa (supervised) + reward model approach
4. **Multiple domains:** News, creative writing, QA, academic text
5. **Decoding strategy control:** Greedy, sampling, with/without repetition penalty

### 7.3 Known Confounds
- **Text length:** GPTZero requires 150+ characters; detection improves with length (Xu & Zubiaga)
- **Code-mixed text:** Detection degrades on mixed code/NL content (QA results in Xu & Zubiaga)
- **Repetition penalty:** Dramatically reduces detectability (up to 32 percentage points in RAID)
- **Domain shift:** Detectors trained on one domain may fail on others (RAID, MAGE)

---

## 8. Gaps in the Literature and Experimental Opportunities

### 8.1 Gaps Our Research Can Address

1. **No systematic study of alignment type vs. detectability.** Papers test SFT+RLHF or just RLHF. No study compares RLHF vs. DPO vs. Constitutional AI vs. RLHF+CAI on the same base model.

2. **No theory connecting alignment strength to TV distance.** Chakraborty et al. prove detection requires TV>0; no paper quantifies how alignment magnitude relates to TV distance.

3. **No cross-family alignment comparison.** Do Llama-RLHF and GPT-RLHF produce similar alignment signatures? ReMoDetect's cross-model transfer suggests yes, but this hasn't been systematically studied.

4. **Limited base model testing.** DetectGPT only tested base models; most detection papers only test aligned models. Direct comparison on the same architecture is rare (Xu & Zubiaga is one of the few).

5. **No formal framing of alignment as implicit watermark.** The phrase appears in DetectGPT's discussion but hasn't been developed into a formal framework.

### 8.2 Proposed Experimental Program

1. **Phase 1 (Distributionial analysis):** Compare token probability distributions between base and aligned model pairs (using RAID data) to quantify TV distance at the empirical level.

2. **Phase 2 (Detection spectrum):** Run DetectGPT, Fast-DetectGPT, and reward model detection across base/SFT/RLHF variants of multiple model families.

3. **Phase 3 (Alignment type comparison):** Compare detectability of different alignment methods (RLHF, DPO, SFT-only, Constitutional) applied to the same base model.

4. **Phase 4 (Formal framework):** Develop the theoretical framework connecting alignment objectives to distributional shifts that enable detection.

---

## 9. Summary of Evidence Chain

The literature supports the following logical chain:

1. **RLHF collapses output diversity** by 57-90% across lexical, semantic, and syntactic measures (Kirk et al., Xu & Zubiaga)

2. **This creates a consistent stylistic mode** -- across-input mode collapse means aligned models produce similar text regardless of prompt (Kirk et al.)

3. **The distributional shift is detectable** by zero-shot methods (Fast-DetectGPT AUROC improves from 0.68 to 0.91 with alignment, Xu & Zubiaga)

4. **Reward models serve as natural detectors** because they encode the very preferences that alignment was optimized toward (ReMoDetect, AUROC 95-99% on aligned vs. 60-75% on base models)

5. **Detection is theoretically guaranteed** for any non-zero distributional shift (Chakraborty et al.), and alignment ensures this shift is non-zero

6. **LLMs "watermark themselves implicitly"** through their learned probability landscapes (DetectGPT), and alignment strengthens this watermark

**Conclusion:** The evidence strongly supports the hypothesis that alignment is the watermark. The alignment training process -- by design -- moves the model's output distribution away from the human text distribution, creating an inherent, detectable signature. This signature is not an artifact or a bug; it is a fundamental consequence of what alignment does: it makes the model's outputs systematically different from human text in ways that serve human preferences but simultaneously enable detection.

---

## References

1. Xu, B. & Zubiaga, A. (2025). Understanding the Effects of RLHF on the Quality and Detectability of LLM-Generated Texts. arXiv:2503.17965.
2. Kirk, R. et al. (2024). Understanding the Effects of RLHF on LLM Generalisation and Diversity. ICLR 2024. arXiv:2310.06452.
3. Lee, J., Tack, J., & Shin, J. (2024). ReMoDetect: Reward Models Recognize Aligned LLM's Generations.
4. Chakraborty, M. et al. (2023). On the Possibilities of AI-Generated Text Detection. arXiv:2304.04736.
5. Mitchell, E. et al. (2023). DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature. ICML 2023. arXiv:2301.11305.
6. Bao, G. et al. (2024). Fast-DetectGPT: Efficient Zero-Shot Detection via Conditional Probability Curvature. ICLR 2024. arXiv:2310.05130.
7. Dugan, L. et al. (2024). RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors. ACL 2024. arXiv:2405.07940.
8. He, X. et al. (2023). MGTBench: Benchmarking Machine-Generated Text Detection. ACM CCS 2024. arXiv:2303.14822.
9. Li, Y. et al. (2024). MAGE: Machine-Generated Text Detection in the Wild. ACL 2024. arXiv:2305.13242.
10. Guo, B. et al. (2023). How Close is ChatGPT to Human Experts? arXiv:2301.07597.
11. Ouyang, L. et al. (2022). Training Language Models to Follow Instructions with Human Feedback. NeurIPS 2022. arXiv:2203.02155.
12. Kirchenbauer, J. et al. (2023). A Watermark for Large Language Models. ICML 2023. arXiv:2301.10226.
13. Kirchenbauer, J. et al. (2024). On the Reliability of Watermarks for Large Language Models. ICLR 2024. arXiv:2306.04634.
14. Dathathri, S. et al. (2024). Scalable Watermarking for Identifying Large Language Model Outputs. Nature.
15. Gehrmann, S. et al. (2019). GLTR: Statistical Detection and Visualization of Generated Text. ACL 2019. arXiv:1906.04043.
16. Ippolito, D. et al. (2020). Automatic Detection of Generated Text is Easiest When Humans are Fooled. ACL 2020. arXiv:1911.00650.
17. Yang, X. et al. (2023). DNA-GPT: Divergent N-Gram Analysis for Training-Free Detection of GPT-Generated Text. arXiv:2305.17359.
18. Tulchinskii, E. et al. (2023). Intrinsic Dimension Estimation for Robust Detection of AI-Generated Texts. arXiv:2306.04723.
19. Sadasivan, V. et al. (2023). Can AI-Generated Text be Reliably Detected? arXiv:2303.11156.
20. Rivera-Soto, R. et al. (2025). AI-generated Text has a Distinct Style.
21. McGovern, H. et al. (2024). Fingerprints of AI-Generated Text.
22. Fraser, K. et al. (2024). Factors Affecting the Detectability of AI-Generated Text.
23. Panickssery, A. et al. (2024). LLM Evaluators Recognize and Favor Their Own Generations.
24. Li, Y. et al. (2024). Predicting vs. Acting: A Trade-off.
25. Bhattacharjee, A. et al. (2023). Fighting Fire with Fire: Can ChatGPT Detect Its Own Text?
26. Hans, A. et al. (2024). Spotting LLMs with Binoculars.
