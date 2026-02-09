# Google Colab Installation Guide

Este documento explica cómo instalar y usar el paquete `intention-collapse` en Google Colab.

## Instalación Rápida (1 cell)

```python
# Clonar repositorio e instalar paquete
!rm -rf intention-collapse-experiments
!git clone -b v2-router-experiments https://github.com/patriciomvera/intention-collapse-experiments.git
!pip install -e /content/intention-collapse-experiments/ -q

# Verificar instalación
from src.router import AdaptiveInferenceRouter, RouteDecision
from src.metrics import compute_intention_entropy
from src.controls import self_consistency_baseline
from src.decoding import constrained_mc_generation

print("✅ Installation successful! Ready to run experiments.")
```

## Notebooks de Ejemplo

### Opción 1: Router Experiment (Recomendado)
Notebook completo con ejemplo funcional del Adaptive Inference Router:

```python
# En Colab, ejecuta:
!git clone -b v2-router-experiments https://github.com/patriciomvera/intention-collapse-experiments.git
%cd intention-collapse-experiments/notebooks
# Abre: colab_router_experiment.ipynb
```

### Opción 2: Experimentos Completos
Para replicar los experimentos del paper:

```python
# En Colab, ejecuta:
!git clone -b v2-router-experiments https://github.com/patriciomvera/intention-collapse-experiments.git
%cd intention-collapse-experiments/notebooks/scaled
# Abre: 01_run_experiments.ipynb
```

## Estructura del Paquete

```python
# Importar módulos principales
from src.router import AdaptiveInferenceRouter, RouteDecision, RouterResult
from src.metrics import compute_intention_entropy, IntentionMetrics
from src.controls import self_consistency_baseline, SelfConsistencyResult
from src.decoding import constrained_mc_generation
```

## Ejemplo Mínimo

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.router import AdaptiveInferenceRouter

# Cargar modelo
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Crear router
router = AdaptiveInferenceRouter(
    model=model,
    tokenizer=tokenizer,
    entropy_threshold=3.0
)

# Ejecutar inferencia adaptativa
result = router.route_and_generate(
    question="What is 2 + 2?",
    max_new_tokens=50
)

print(f"Route: {result.route_taken}")
print(f"Entropy: {result.intention_entropy:.3f}")
print(f"Answer: {result.extracted_answer}")
```

## Dependencias

El comando `pip install -e .` instala automáticamente:

- `torch>=2.0.0`
- `transformers>=4.36.0`
- `numpy>=1.24.0`
- `scikit-learn>=1.3.0`
- `datasets>=2.14.0`
- `matplotlib>=3.7.0`
- `seaborn>=0.12.0`

## Troubleshooting

### Error: "No module named 'src'"

**Causa:** El paquete no está instalado correctamente.

**Solución:**
```python
!pip install -e /content/intention-collapse-experiments/
```

### Error: "No module named 'torch'"

**Causa:** Las dependencias no se instalaron.

**Solución:**
```python
# Instalar con dependencias explícitamente
!pip install -e /content/intention-collapse-experiments/
```

### Error: "attempted relative import with no known parent package"

**Causa:** Estás intentando ejecutar archivos de src/ directamente.

**Solución:** Importa desde el paquete instalado:
```python
# ✅ Correcto
from src.router import AdaptiveInferenceRouter

# ❌ Incorrecto
%run src/router/adaptive_router.py
```

### Verificar que todo funciona

Ejecuta este script de test:

```python
import sys
import subprocess

# Test 1: Verificar instalación del paquete
try:
    import src
    print(f"✅ Package 'src' installed (version {src.__version__})")
except ImportError as e:
    print(f"❌ Cannot import 'src': {e}")
    sys.exit(1)

# Test 2: Verificar imports críticos
try:
    from src.router import AdaptiveInferenceRouter, RouteDecision
    print("✅ src.router imports OK")
except ImportError as e:
    print(f"❌ src.router import failed: {e}")
    sys.exit(1)

try:
    from src.metrics import compute_intention_entropy
    print("✅ src.metrics imports OK")
except ImportError as e:
    print(f"❌ src.metrics import failed: {e}")
    sys.exit(1)

try:
    from src.controls import self_consistency_baseline
    print("✅ src.controls imports OK")
except ImportError as e:
    print(f"❌ src.controls import failed: {e}")
    sys.exit(1)

try:
    from src.decoding import constrained_mc_generation
    print("✅ src.decoding imports OK")
except ImportError as e:
    print(f"❌ src.decoding import failed: {e}")
    sys.exit(1)

print("\n✅ ALL TESTS PASSED - Ready to run experiments!")
```

## Recursos

- **Quick Start Notebook:** `notebooks/colab_router_experiment.ipynb`
- **Full Experiments:** `notebooks/scaled/01_run_experiments.ipynb`
- **Documentation:** [README.md](README.md)
- **Paper:** [arXiv:2601.01011](https://arxiv.org/abs/2601.01011)

## Soporte

Si encuentras problemas:

1. Verifica que estás usando Python 3.10+
2. Verifica que el branch es `v2-router-experiments`
3. Ejecuta el script de verificación arriba
4. Reporta issues en: https://github.com/patriciomvera/intention-collapse-experiments/issues
