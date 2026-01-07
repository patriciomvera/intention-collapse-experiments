# Pilot Study: Initial Validation Experiment

This directory contains the initial pilot experiment that validated the Intention Collapse framework on a small scale before scaling to the full research agenda.

## Experiment Details

### Dataset
- **Benchmark**: GSM8K (Grade School Math 8K)
- **Split**: Test set
- **Sample size**: 200 problems (randomly selected, seed=42)
- **Problem type**: Elementary school math word problems

### Model
- **Name**: Mistral-7B-Instruct-v0.3
- **Quantization**: 4-bit (NF4)
- **Extraction layers**: 27-31 (last 5 layers)
- **Hardware**: Google Colab (T4/L4 GPU)

### Conditions
1. **Baseline**: Direct answer, greedy decoding, max 50 tokens
2. **Enhanced (CoT)**: Chain-of-thought prompting, max 512 tokens
3. **Babble**: Length-matched control, unstructured generation

### Metrics Evaluated
- **Intention Entropy** H_int(I): Pre-collapse next-token entropy
- **Effective Dimensionality** dim_eff(I): PCA-based dimensionality (global & per-layer)
- **Latent Recoverability** Recov(I;Z): Linear probe AUROC for correctness prediction

## Key Results

| Metric | Baseline | CoT | Babble |
|--------|----------|-----|--------|
| Accuracy | 5.5% | 53.0% | N/A |
| H_int (bits) | 1.42 | 0.37 | 0.96 |
| dim_eff (global) | 2.43 | 2.85 | 2.25 |
| Probe AUROC | 0.56 [0.39-0.75] | 0.65 [0.57-0.72] | N/A |

**Key findings:**
- CoT dramatically improves accuracy (5.5% → 53%)
- CoT reduces intention entropy by ~73% (1.42 → 0.37 bits)
- CoT shows higher global dimensionality despite producing structured reasoning
- Probe recovers latent correctness information in CoT (AUROC=0.65) but collapses to majority class in baseline
- Entropy has minimal per-item predictive power (|r| ≈ 0.06)

## Notebooks

### `01_pilot_gsm8k.ipynb`
The complete pilot experiment implementation:
1. Setup and model loading (Mistral-7B, 4-bit)
2. Dataset loading (200 GSM8K problems)
3. Activation extraction with hooks
4. Metric computation (entropy, dimensionality, probes)
5. Results aggregation and visualization

## Running the Pilot

### Google Colab (Recommended)
```python
# 1. Open notebook in Colab
# 2. Runtime > Change runtime type > T4 GPU
# 3. Run all cells
```

### Local Execution
```bash
# Requires GPU with ≥12GB VRAM
jupyter notebook notebooks/pilot/01_pilot_gsm8k.ipynb
```

## Historical Context

This pilot study was conducted as a proof-of-concept to:
1. Validate that intention metrics are implementable and measurable
2. Establish baseline patterns (CoT enriches I, babble does not)
3. Identify methodological issues (data leakage in probes, symbolic evaluation needs)
4. Estimate compute requirements for scaling

The pilot directly informed the scaled experiments in `notebooks/scaled/`, which address the methodological issues and expand to 3 models × 3 benchmarks.

## Citation

If you use this pilot study, please cite:

```bibtex
@article{vera2025intention,
  title={Intention Collapse: A Unified Framework for Understanding Reasoning in Language Models},
  author={Vera, P. M.},
  journal={arXiv preprint},
  year={2025}
}
```

## References

- GSM8K: Cobbe et al. (2021) - https://arxiv.org/abs/2110.14168
- Mistral-7B: Jiang et al. (2023) - https://arxiv.org/abs/2310.06825
- Chain-of-Thought: Wei et al. (2022) - https://arxiv.org/abs/2201.11903
