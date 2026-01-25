# Contributing to Intention Collapse Experiments

Thank you for your interest in contributing! This project aims to empirically validate the Intention Collapse framework for understanding reasoning in Large Language Models.

## How to Contribute

### Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include your environment details (Python version, GPU, etc.)
- For bugs, include steps to reproduce and error messages

### Code Contributions

1. **Fork** the repository
2. **Create a branch** for your feature (`git checkout -b feature/amazing-feature`)
3. **Make your changes** following our code style
4. **Test** your changes thoroughly
5. **Commit** with clear messages following conventional commits
6. **Push** to your branch (`git push origin feature/amazing-feature`)
7. **Open a Pull Request**

## Conventions

### Git Commits

We use [Conventional Commits](https://www.conventionalcommits.org/). Format:

```
<type>: <description>

[optional body]
```

Types:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

Examples:
```bash
git commit -m "feat: Add MATH benchmark support"
git commit -m "fix: Correct entropy calculation for edge cases"
git commit -m "docs: Update installation instructions"
```

### Python Code Style

- Follow PEP 8 for Python code
- Use type hints where possible
- Document functions with docstrings (Google style)
- Keep lines under 88 characters (Black formatter compatible)
- Use pytest for tests

### LaTeX (Paper)

- Use `\parencite{}` for citations
- Use BibTeX for references
- Paper source syncs with Overleaf

## Areas for Contribution

We especially welcome contributions in these areas:

### High Priority
- [ ] Implement Experiment 4.2 (State-dependent collapse variability)
- [ ] Implement Experiment 4.3 (Latent knowledge recovery with quirky models)
- [ ] Add MATH benchmark support
- [ ] Compute-matched controls (length-matched structured reasoning)

### Medium Priority
- [ ] Option-normalized entropy for MCQ tasks
- [ ] Add statistical significance tests (McNemar on per-item outcomes)
- [ ] Support for multi-GPU setups
- [ ] Experiment tracking with W&B
- [ ] Cross-modal validation (vision-language models)

### Documentation
- [ ] Add more usage examples
- [ ] Create video tutorials
- [ ] Improve inline code documentation

### Completed
- [x] 3x3 experimental matrix (3 models x 3 benchmarks)
- [x] Support for Mistral-7B, LLaMA-3.1-8B, Qwen-2.5-7B
- [x] GSM8K, ARC-Challenge, AQUA-RAT benchmarks
- [x] Probe robustness analysis
- [x] Cross-regime transfer analysis

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/intention-collapse-experiments.git
cd intention-collapse-experiments

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install pytest black flake8  # Development tools

# Run tests
pytest tests/

# Format code
black src/
```

## Questions?

Open a Discussion on GitHub or reach out through Issues.

Thank you for contributing!
