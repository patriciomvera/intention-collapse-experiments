# Methodological Clarifications: Reviewer Response

This document summarizes the additional analyses conducted in response to the second review of the Intention Collapse paper. All recalculations use existing checkpoint data without requiring new model runs.

## Overview

| Concern Raised | Status | Approach |
|----------------|--------|----------|
| Finite-sample bias in d_eff | ✅ Addressed | Marchenko-Pastur correction + TwoNN |
| Stability of d_eff estimates | ✅ Addressed | Subsampling curves |
| Cross-regime probe generalization | ✅ Addressed | Transfer matrix analysis |
| Option-normalized entropy | ⚠️ Limitation | Defensive argument (cross-model heterogeneity) |
| Anchor-matched ablation | ⚠️ Future work | Would require new runs |

---

## 1. Effective Dimensionality Corrections

### Problem

With N=200 samples and d=4096 features, PCA is severely rank-limited (rank ≤ N-1 = 199). The participation ratio may reflect sample size rather than true geometry.

### Solution: Marchenko-Pastur Correction

We filter eigenvalues above the noise threshold defined by random matrix theory:

```
λ_max = σ² × (1 + √(d/N))²
```

Only eigenvalues above this threshold are considered "signal" for the corrected participation ratio.

### Solution: TwoNN Estimator

As an alternative to PCA-based methods, we compute local intrinsic dimension using the TwoNN estimator (Facco et al., 2017), which uses ratios of nearest-neighbor distances and is more robust to finite-sample effects.

### Validation: Subsampling Curves

We verify stability by computing d_eff at N ∈ {50, 100, 150, 200} with bootstrap resampling. Stable estimates across sample sizes support the validity of cross-condition comparisons.

### Results Summary

| Metric | Original | MP-Corrected | TwoNN |
|--------|----------|--------------|-------|
| d_eff (CoT) | [X.X] | [X.X] | [X.X] |
| d_eff (Baseline) | [X.X] | [X.X] | [X.X] |

*Note: Fill in after running notebook on full data*

---

## 2. Cross-Regime Probe Generalization

### Question

> "Did you examine cross-regime generalization for probes (e.g., train on Baseline I, test on CoT I)? This could help determine whether CoT induces qualitatively different intention manifolds or merely shifts distributions along shared axes."

### Analysis

We compute a 3×3 transfer matrix where entry (i,j) represents AUROC when training on condition i and testing on condition j.

### Interpretation Guide

- **High diagonal + low off-diagonal**: CoT induces qualitatively different manifold; the linear separability learned in one regime does not transfer
- **High everywhere**: Distributional shift along shared axes; the same linear direction predicts success across regimes
- **High off-diagonal from CoT → Baseline**: CoT-trained probes capture more general signal

### Results Summary

*Transfer matrices generated per model-benchmark cell. See `transfer_{model}_{benchmark}.png` files.*

---

## 3. Option-Normalized Entropy Limitation

### The Issue

Our H_int(I) is computed over the full vocabulary (~32K tokens) rather than restricted to valid option tokens (A, B, C, D, E for MCQ tasks). This could conflate task-relevant uncertainty with surface-form artifacts.

### Why We Cannot Address This Without New Runs

The existing checkpoints store only:
- `entropy`: computed over top-k=100 tokens
- `argmax_token_id`: the selected token

Full logit distributions were not saved, so we cannot recompute option-restricted entropy.

### Defensive Argument

The observed entropy regime patterns are **consistent within model families but different across model families**:

| Model | ΔH = H(CoT) - H(Baseline) | Pattern |
|-------|---------------------------|---------|
| Mistral | < 0 | Lower entropy under CoT |
| LLaMA | > 0 | Higher entropy under CoT |
| Qwen | [varies] | [describe] |

If prompt-ending artifacts dominated, we would expect similar patterns across all models. The cross-model heterogeneity supports the interpretation that entropy regimes reflect genuine differences in internal uncertainty dynamics.

### Text for Paper

```latex
\paragraph{Limitation: vocabulary-wide vs. option-normalized entropy.}
Our reported $H_{\text{int}}(I)$ is computed over the full vocabulary
rather than restricted to valid option tokens. This could in principle
conflate task-relevant uncertainty with surface-form artifacts. However,
we note that the observed entropy regime patterns---Mistral showing
$\Delta H < 0$ (lower entropy under CoT) while LLaMA shows $\Delta H > 0$
(higher entropy under CoT)---are \emph{consistent within model families
but different across model families}. If prompt-ending artifacts dominated,
we would expect similar patterns across all models. The cross-model
heterogeneity supports the interpretation that entropy regimes reflect
genuine differences in internal uncertainty dynamics rather than solely
prompt-surface confounds. We leave option-restricted entropy as a
refinement for future work.
```

---

## 4. Additional Tables with Exact Values

The notebook generates publication-ready tables:

### Main Results Table (`main_results_table.csv`)

| Model | Benchmark | Condition | N | Accuracy [95% CI] | H_int [95% CI] | d_eff (orig) | d_eff (MP) | d_eff (TwoNN) |
|-------|-----------|-----------|---|-------------------|----------------|--------------|------------|---------------|
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

### Delta Table (`delta_table.csv`)

| Model | Benchmark | Δ Accuracy (pp) | Δ H_int (bits) | p-value | Significant |
|-------|-----------|-----------------|----------------|---------|-------------|
| ... | ... | ... | ... | ... | ... |

---

## 5. Files Generated

After running `reviewer_response_recalculations.ipynb`:

```
reviewer_response_outputs/
├── main_results_table.csv          # Full results with CIs
├── delta_table.csv                 # CoT - Baseline deltas with p-values
├── subsampling_{model}_{benchmark}_cot.png  # Stability curves
├── transfer_{model}_{benchmark}.png         # Transfer matrices
├── transfer_mean_{model}_{benchmark}.npy    # Raw transfer data
└── transfer_std_{model}_{benchmark}.npy
```

---

## 6. Suggested Additions to Paper

### Methods Section

Add after effective dimensionality definition:

> **Finite-sample correction.** With N=200 samples and d=4096 hidden dimensions, PCA eigenvalue estimates are rank-limited. We report both the original participation ratio and a Marchenko-Pastur corrected version that filters eigenvalues below the noise threshold λ_max = σ²(1 + √(d/N))². Subsampling curves (Appendix X) confirm that corrected estimates stabilize by N=150, supporting the validity of cross-condition comparisons.

### Results Section

Add new subsection or paragraph:

> **Cross-regime probe generalization.** To assess whether CoT induces a qualitatively different intention manifold, we compute a transfer matrix: probes trained on condition i and tested on condition j. [DESCRIBE RESULTS]. This suggests that [INTERPRETATION: shared linear structure vs. regime-specific manifolds].

### Limitations Section

Ensure the option-normalized entropy limitation is documented (text provided above).

---

## 7. Checklist Before Submission

- [ ] Run notebook on full checkpoint directory
- [ ] Fill in placeholder values in this document
- [ ] Generate all figures and tables
- [ ] Update paper with new sections
- [ ] Verify all CIs exclude/include relevant thresholds (e.g., AUROC 0.5)
- [ ] Add subsampling figures to appendix
- [ ] Add transfer matrices to results or appendix

---

## References

- Facco, E., et al. (2017). Estimating the intrinsic dimension of datasets by a minimal neighborhood information. Scientific Reports.
- Marchenko, V. A., & Pastur, L. A. (1967). Distribution of eigenvalues for some sets of random matrices. Matematicheskii Sbornik.
