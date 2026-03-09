# Project Reorganization Proposal

## Current Issues

### 1. Conflicting Content Organization
- **`experiments/`**: Contains NEW adaptive router experiments
- **`notebooks/`**: Contains ORIGINAL intention collapse research + NEW Colab router demos
- Confusion about which experiments are for which research direction

### 2. Development Artifacts in Main Branch
Files that should NOT be in main:
- `TASK_3_SUMMARY.md` (Spanish, development notes)
- `TASK_4_SUMMARY.md` (Spanish, development notes)
- `TASK_5_SUMMARY.md` (Spanish, development notes)
- `TASK_6_SUMMARY.md` (Spanish, development notes)
- `PACKAGE_FIX_SUMMARY.md` (development documentation)
- `PROJECT_COMPLETION_SUMMARY.md` (development documentation)
- `QUICK_TEST.md` (development documentation)

### 3. Language Inconsistency
- Several development files in Spanish
- Need to verify all code and comments are in English

## Proposed New Structure

```
intention-collapse-experiments/
├── README.md                           # Main project overview (UPDATED)
├── CONTRIBUTING.md                     # Contribution guidelines
├── COLAB_INSTALL.md                    # Colab setup instructions
├── LICENSE
├── .gitignore
├── pyproject.toml                      # Package configuration
├── setup.py                            # Package setup
├── requirements.txt                    # Dependencies
│
├── configs/
│   └── experiment_config.yaml          # Experiment hyperparameters
│
├── docs/
│   ├── README.md                       # Documentation index
│   ├── paper/                          # Original research paper
│   │   └── Intention_Collapse.pdf
│   ├── constrained_decoding.md         # Technical docs
│   ├── option_normalized_entropy.md    # Technical docs
│   └── self_consistency.md             # Technical docs
│
├── examples/                           # Standalone demo scripts
│   ├── README.md                       # NEW: Examples overview
│   ├── router_demo.py                  # Adaptive router quick demo
│   ├── option_normalized_demo.py       # Option-normalized entropy demo
│   ├── self_consistency_demo.py        # Self-consistency demo
│   └── constrained_decoding_demo.py    # Constrained decoding demo
│
├── notebooks/                          # REORGANIZED: Clear separation by purpose
│   ├── README.md                       # NEW: Notebooks overview
│   │
│   ├── original_research/              # RENAMED from pilot/ and scaled/
│   │   ├── README.md                   # Overview of intention collapse research
│   │   ├── pilot/                      # Initial validation
│   │   │   ├── README.md
│   │   │   └── 01_pilot_gsm8k.ipynb
│   │   └── scaled/                     # Full 3x3 experiments
│   │       ├── README.md
│   │       ├── METHODOLOGICAL_CLARIFICATIONS.md
│   │       ├── 01_run_experiments.ipynb
│   │       ├── 02_consolidate_results.ipynb
│   │       └── reviewer_response_recalculations.ipynb
│   │
│   └── adaptive_router/                # NEW: Adaptive router experiments
│       ├── README.md                   # NEW: Router experiments overview
│       ├── colab_demo.ipynb            # RENAMED from colab_router_experiment.ipynb
│       └── full_experiment.ipynb       # MOVED from experiments/router_experiment.ipynb
│
├── scripts/                            # NEW: Standalone Python scripts
│   ├── README.md                       # Scripts overview
│   ├── run_router_experiment.py        # MOVED from experiments/
│   ├── colab_setup.py                  # Colab automated setup
│   └── verify_colab_setup.py           # Colab setup verification
│
├── src/                                # Source code (unchanged structure)
│   ├── __init__.py
│   ├── activation_hooks.py
│   ├── checkpoint_utils.py
│   ├── data_utils.py
│   ├── metrics.py
│   ├── probing.py
│   ├── shared_utils.py
│   ├── visualization.py
│   ├── controls/
│   │   ├── __init__.py
│   │   └── self_consistency.py
│   ├── decoding/
│   │   ├── __init__.py
│   │   └── constrained.py
│   └── router/
│       ├── __init__.py
│       ├── README.md
│       └── adaptive_router.py
│
├── tests/                              # Test suite
│   ├── README.md
│   ├── test_imports.py
│   ├── test_option_normalized_entropy.py
│   └── test_package_structure.py
│
└── results/                            # Experiment results (gitignored)
    ├── original_research/              # Results from intention collapse experiments
    └── adaptive_router/                # Results from router experiments
```

## Key Changes

### 1. Clear Research Separation

**Original Research (Intention Collapse Framework)**
- Location: `notebooks/original_research/`
- Contents: pilot + scaled experiments from the paper
- Purpose: Validate intention collapse framework across 3 models × 3 benchmarks

**New Research (Adaptive Router)**
- Location: `notebooks/adaptive_router/` + `scripts/`
- Contents: Router experiments using intention entropy for inference routing
- Purpose: Apply intention entropy for practical routing decisions

### 2. File Moves and Renames

| Current Path | New Path | Reason |
|-------------|----------|---------|
| `notebooks/pilot/` | `notebooks/original_research/pilot/` | Clarify research lineage |
| `notebooks/scaled/` | `notebooks/original_research/scaled/` | Clarify research lineage |
| `notebooks/colab_router_experiment.ipynb` | `notebooks/adaptive_router/colab_demo.ipynb` | More descriptive name |
| `notebooks/colab_quick_test.ipynb` | `notebooks/adaptive_router/quick_test.ipynb` | Better organization |
| `experiments/router_experiment.ipynb` | `notebooks/adaptive_router/full_experiment.ipynb` | Consolidate notebooks |
| `experiments/run_router_experiment.py` | `scripts/run_router_experiment.py` | Dedicated scripts folder |
| `colab_setup.py` | `scripts/colab_setup.py` | Dedicated scripts folder |
| `verify_colab_setup.py` | `scripts/verify_colab_setup.py` | Dedicated scripts folder |

### 3. Files to DELETE

Development artifacts that don't belong in main:
- `TASK_3_SUMMARY.md`
- `TASK_4_SUMMARY.md`
- `TASK_5_SUMMARY.md`
- `TASK_6_SUMMARY.md`
- `PACKAGE_FIX_SUMMARY.md`
- `PROJECT_COMPLETION_SUMMARY.md`
- `QUICK_TEST.md`
- `experiments/` directory (after moving contents)

### 4. New README Files Needed

Create these new overview files:
- `notebooks/README.md` - Overview of all notebooks
- `notebooks/original_research/README.md` - Intention collapse research overview
- `notebooks/adaptive_router/README.md` - Router experiments overview
- `scripts/README.md` - Scripts usage guide
- `examples/README.md` - Examples overview

### 5. Updates to Existing Files

**Main README.md** - Update structure section to reflect new organization:
```markdown
## Repository Structure

### Core Components
- `src/` - Source code for intention metrics and adaptive routing
- `notebooks/` - Jupyter notebooks for experiments
  - `original_research/` - Intention Collapse framework validation (Paper)
  - `adaptive_router/` - Adaptive routing experiments (New)
- `examples/` - Standalone demo scripts
- `scripts/` - Utility scripts for Colab and batch execution
- `docs/` - Documentation and paper

### Quick Start
- **Original Research**: See `notebooks/original_research/README.md`
- **Adaptive Router**: See `notebooks/adaptive_router/README.md`
- **Examples**: See `examples/README.md`
```

**COLAB_INSTALL.md** - Update paths:
- Change references from `experiments/` to `notebooks/adaptive_router/`
- Change script paths to `scripts/`

**All READMEs in subdirectories** - Update internal path references

## Implementation Plan

### Phase 1: Preparation
1. Create new directory structure (empty folders)
2. Create new README.md files for each section
3. Review all files for Spanish content

### Phase 2: File Moves
1. Move notebooks to new locations
2. Move scripts to `scripts/` folder
3. Move examples documentation

### Phase 3: Cleanup
1. Delete development artifact files (TASK_*.md, etc.)
2. Remove empty `experiments/` directory
3. Update all cross-references in READMEs

### Phase 4: Verification
1. Test all imports still work
2. Verify Colab notebooks run with new paths
3. Check all documentation links are valid
4. Run test suite

### Phase 5: Language Audit
1. Grep for Spanish comments in code
2. Translate any remaining Spanish content
3. Verify all user-facing documentation is in English

## Backward Compatibility Considerations

### Import Paths (No Change)
All Python imports remain the same:
```python
from src.router import AdaptiveInferenceRouter
from src.metrics import compute_intention_entropy
# etc.
```

### Colab Setup (Minor Update)
Update installation instructions to use new paths:
```python
# Old
!jupyter notebook experiments/router_experiment.ipynb

# New
!jupyter notebook notebooks/adaptive_router/full_experiment.ipynb
```

### GitHub Links
Update any absolute GitHub links in documentation to reflect new paths.

## Benefits

1. **Clarity**: Clear separation between original research and new router work
2. **Maintainability**: Development artifacts removed from main branch
3. **Usability**: Users can quickly find relevant experiments
4. **Professionalism**: Clean, well-organized repository for publication
5. **Consistency**: All content in English
6. **Scalability**: Clear structure for future additions

## Questions for Approval

Before proceeding, please confirm:

1. ✅ Approve the overall structure?
2. ✅ Agree with `original_research/` and `adaptive_router/` naming?
3. ✅ Confirm deletion of TASK_*.md files?
4. ✅ Any additional folders or files to reorganize?

---

**Next Steps**: Once approved, I will:
1. Execute the reorganization
2. Update all documentation
3. Verify everything works
4. Create a commit with the reorganization
