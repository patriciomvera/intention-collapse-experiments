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
5. **Commit** with clear messages (`git commit -m 'Add amazing feature'`)
6. **Push** to your branch (`git push origin feature/amazing-feature`)
7. **Open a Pull Request**

### Code Style

- Follow PEP 8 for Python code
- Use type hints where possible
- Document functions with docstrings (Google style)
- Keep lines under 88 characters (Black formatter compatible)

### Areas for Contribution

We especially welcome contributions in these areas:

#### High Priority
- [ ] Implement Experiment 4.2 (State-dependent collapse variability)
- [ ] Implement Experiment 4.3 (Latent knowledge recovery)
- [ ] Add support for Llama-3 models
- [ ] Add MATH benchmark support

#### Medium Priority
- [ ] Improve visualization options
- [ ] Add statistical significance tests
- [ ] Support for multi-GPU setups
- [ ] Experiment tracking with W&B

#### Documentation
- [ ] Add more examples
- [ ] Improve docstrings
- [ ] Create video tutorials

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/intention-collapse-experiments.git
cd intention-collapse-experiments

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -r requirements.txt
pip install pytest black flake8  # Development tools

# Run tests
pytest tests/

# Format code
black src/
```

## Questions?

Open a Discussion on GitHub or reach out through Issues.

Thank you for contributing! 🙏
