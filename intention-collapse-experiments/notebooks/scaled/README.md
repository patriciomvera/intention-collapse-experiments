# Scaled Experiments: Full Research Agenda (3×3 Design)

This directory contains the scaled experimental implementation addressing reviewer feedback and expanding the Intention Collapse validation to multiple models and benchmarks.

## Experiment Design

### 3×3 Matrix: Models × Benchmarks

| | GSM8K | MATH | ARC-Challenge |
|---|---|---|---|
| **Mistral-7B** | ✓ | ✓ | ✓ |
| **Llama-3.1-8B** | ✓ | ✓ | ✓ |
| **Qwen-2.5-7B** | ✓ | ✓ | ✓ |

**Total**: 9 experiment runs × 3 conditions = 27 experimental configurations

### Models

1. **Mistral-7B-Instruct-v0.3**
   - Layers: 32
   - Extraction: Layers 27-31
   - Context: 32K tokens

2. **Llama-3.1-8B-Instruct**
   - Layers: 32
   - Extraction: Layers 27-31
   - Context: 128K tokens

3. **Qwen-2.5-7B-Instruct**
   - Layers: 28
   - Extraction: Layers 23-27
   - Context: 128K tokens

### Benchmarks

1. **GSM8K** (Grade School Math)
   - Problems: 200 (consistent across models)
   - Task: Elementary arithmetic reasoning
   - Evaluation: Exact numerical match

2. **MATH** (Competition Math)
   - Problems: 200 (consistent across models)
   - Task: High school/competition mathematics
   - Evaluation: Symbolic equivalence (sympy)

3. **ARC-Challenge** (AI2 Reasoning Challenge)
   - Problems: 200 (consistent across models)
   - Task: Science multiple choice questions
   - Evaluation: Exact letter match (A/B/C/D)

### Conditions

All experiments run three conditions:

1. **Baseline**: Direct answer prompting (max 50 tokens, greedy)
2. **Enhanced (CoT)**: Chain-of-thought prompting (max 512 tokens, greedy)
3. **Babble**: Length-matched control (max 200 tokens, greedy)

### Intention Metrics

Three model-agnostic metrics quantify the pre-collapse state I:

1. **H_int(I)**: Entropy of next-token distribution (bits)
2. **dim_eff(I)**: Effective dimensionality via PCA (global + per-layer)
3. **Recov(I;Z)**: Linear probe AUROC for correctness prediction

## Architecture: Single Source of Truth

All experiment logic lives in `src/shared_utils.py` v3.0 (1347 lines):

```python
import shared_utils as U

# Everything needed:
U.load_problems()            # Dataset loading
U.ActivationExtractor()      # Activation capture
U.run_single_problem()       # Inference + metrics
U.train_recoverability_probe()  # Probe training
U.save_experiment_results()  # Persistence
```

**Key improvements from pilot:**
- ✅ No data leakage (Pipeline in probes, train/test by item)
- ✅ Symbolic evaluation for MATH (sympy)
- ✅ Robust answer extraction (last match, multiple patterns)
- ✅ Memory-efficient (activation extraction per-problem)
- ✅ Reproducible (saved dataset indices, version tracking)

## Notebooks

### 1. `01_run_experiments.ipynb` - Execution

**Purpose**: Run a single experiment (one model × one benchmark)

**Workflow**:
1. Setup: Mount Drive, install dependencies, load shared_utils.py
2. Configuration: Set `MODEL_FAMILY` and `BENCHMARK` (only 2 variables!)
3. Load model and data (consistent indices across all runs)
4. Run 3 conditions (baseline, enhanced, babble)
5. Compute metrics (entropy, dimensionality, probe AUROC)
6. Save results to Google Drive
7. Quick visualization

**Usage**:
```python
# Configure these two lines:
MODEL_FAMILY = 'mistral'  # or 'llama', 'qwen'
BENCHMARK = 'gsm8k'       # or 'math', 'arc'

# Then: Runtime > Run all
```

**Repeat 9 times** for all combinations.

**Compute estimate**: ~30-45 min per run on H100 (4-bit quantization)

### 2. `02_consolidate_results.ipynb` - Analysis

**Purpose**: Consolidate all 9 experiments and generate paper outputs

**Run AFTER all 9 experiments complete**

**Workflow**:
1. Load all results from Drive
2. Verify dataset consistency (same problems across models)
3. Build summary DataFrame
4. Generate LaTeX table for paper
5. Create publication-quality figures:
   - Accuracy heatmap (3×3 grid)
   - Entropy distributions by condition
   - Probe AUROC with confidence intervals
   - Dimensionality comparisons (global vs. per-layer)
6. Statistical tests (Wilcoxon, bootstrap CIs)
7. Export figures and tables to Drive

**Output**: Camera-ready figures and tables for paper submission

## Execution Guide

### Step-by-step: Running All 9 Experiments

**Setup (once)**:
1. Upload `shared_utils.py` to Google Drive: `/MyDrive/intention_collapse_v3/`
2. Create Colab secrets: Add `HF_TOKEN` (HuggingFace API token)
3. Open `01_run_experiments.ipynb` in Colab

**For each of 9 combinations**:

```python
# Edit these two lines in Section 2:
MODEL_FAMILY = 'mistral'  # Change: mistral → llama → qwen
BENCHMARK = 'gsm8k'       # Change: gsm8k → math → arc

# Then:
# - Runtime > Restart runtime
# - Runtime > Run all
# - Wait ~40 minutes
# - Results auto-save to Drive
```

**Checklist**:
- [ ] mistral + gsm8k
- [ ] mistral + math
- [ ] mistral + arc
- [ ] llama + gsm8k
- [ ] llama + math
- [ ] llama + arc
- [ ] qwen + gsm8k
- [ ] qwen + math
- [ ] qwen + arc

**After all 9**:
1. Open `02_consolidate_results.ipynb`
2. Runtime > Run all
3. Download figures and tables from Drive

**Total compute**: ~6-7 hours on Colab H100

## Results Location

All results save to Google Drive:

```
/MyDrive/intention_collapse_v3/
├── splits/                          # Dataset indices (reproducibility)
│   ├── gsm8k_seed42_n200.json
│   ├── math_seed42_n200.json
│   └── arc_seed42_n200.json
├── mistral_gsm8k_results.json      # Detailed results (JSON)
├── mistral_gsm8k_activations.npz   # Activations (NumPy compressed)
├── mistral_gsm8k_summary.png       # Quick viz
├── [... 8 more model-benchmark pairs ...]
└── paper_outputs/                   # From consolidation notebook
    ├── accuracy_heatmap.pdf
    ├── entropy_distributions.pdf
    ├── probe_auroc.pdf
    ├── results_table.tex
    └── supplementary_tables.tex
```

## Reproducibility Checklist

To ensure full reproducibility:

- [x] Dataset indices saved and reused across models
- [x] Random seeds fixed (seed=42 everywhere)
- [x] Code version tracked (v3.0)
- [x] Model versions explicit (HuggingFace model IDs)
- [x] Environment info logged (torch, transformers, cuda versions)
- [x] Train/test splits by item (no cross-contamination)
- [x] Probe uses Pipeline (no data leakage)
- [x] All configs saved with results

## Methodological Improvements

Addressing reviewer feedback from pilot:

| Issue | Solution |
|-------|----------|
| Data leakage in probes | Pipeline with per-fold normalization |
| MATH symbolic evaluation | Sympy simplification + equivalence check |
| Answer extraction ambiguity | Last match, multiple regex patterns |
| Prompt confounds | Fixed templates across models |
| Memory efficiency | Per-problem extraction, no batch stacking |
| Reproducibility | Saved indices, version tracking, environment logs |

## Expected Results (Hypotheses)

Based on the Intention Collapse framework, we expect:

1. **Accuracy**: CoT > Baseline across all models and benchmarks
2. **Entropy**: CoT < Baseline (more decided intentions)
3. **Dimensionality**: CoT ≥ Baseline (richer pre-collapse state)
4. **Recoverability**: CoT probe AUROC > Baseline (latent information preserved)
5. **Cross-benchmark consistency**: Patterns replicate across GSM8K, MATH, ARC
6. **Cross-model consistency**: Different models show similar intention collapse signatures

## Troubleshooting

**Out of memory**:
- Reduce `subset_size` in config (e.g., 100 instead of 200)
- Use Colab Pro (more RAM)

**Model not loading**:
- Verify HF_TOKEN is set correctly
- Check model name in `MODEL_CONFIGS`

**Results not saving**:
- Verify Drive is mounted: `os.path.exists(DRIVE_PATH)`
- Check write permissions

**Probes failing**:
- Verify balanced classes (not all same label)
- Check activation shapes are consistent

## Citation

If you use these scaled experiments:

```bibtex
@article{vera2025intention,
  title={Intention Collapse: A Unified Framework for Understanding Reasoning in Language Models},
  author={Vera, P. M.},
  journal={arXiv preprint},
  year={2025},
  note={Scaled experiments: 3 models × 3 benchmarks}
}
```

## Next Steps

After completing these experiments:

1. **Paper revision**: Incorporate results into manuscript
2. **Extended analysis**: Cross-model comparisons, failure modes
3. **Experiment 4.2**: State-dependent temperature policies
4. **Experiment 4.3**: Latent recovery with quirky models

See main paper for full research agenda.
