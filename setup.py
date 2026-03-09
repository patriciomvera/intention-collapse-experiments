from setuptools import setup, find_packages

setup(
    name="intention-collapse",
    version="2.0.0",
    packages=find_packages(include=["src", "src.*"]),
    package_dir={"src": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "datasets>=2.14.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
)