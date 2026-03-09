# Quick Test Checklist

Usa esta checklist para verificar rápidamente que todo funciona.

## ✅ Checklist Local (Pre-commit)

```bash
# 1. Test estructura del paquete
python tests/test_package_structure.py
# Espera: [SUCCESS] All package structure tests passed!

# 2. Verificar que el paquete se puede importar
python -c "import src; print(f'✅ Package version: {src.__version__}')"
# Espera: ✅ Package version: 2.0.0

# 3. Verificar pyproject.toml
python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"
# Espera: Sin output (éxito silencioso)

# ✅ SI TODO PASA: Listo para commit/push
```

## ✅ Checklist Google Colab

Copia y pega esto en una cell de Colab:

```python
# === SETUP ===
!rm -rf intention-collapse-experiments
!git clone -b v2-router-experiments https://github.com/patriciomvera/intention-collapse-experiments.git
!pip install -e /content/intention-collapse-experiments/ -q

# === VERIFICACIÓN ===
!python /content/intention-collapse-experiments/verify_colab_setup.py

# === TEST IMPORTS ===
from src.router import AdaptiveInferenceRouter, RouteDecision
from src.metrics import compute_intention_entropy
from src.controls import self_consistency_baseline
from src.decoding import constrained_mc_generation

print("\n" + "="*50)
print("✅ ALL IMPORTS SUCCESSFUL")
print("✅ Ready to run experiments!")
print("="*50)
```

**Espera:** Todos los tests pasan y el mensaje "✅ Ready to run experiments!" aparece.

## ✅ Checklist Notebook

Prueba el notebook de ejemplo:

1. Abre Google Colab
2. Carga: `notebooks/colab_router_experiment.ipynb`
3. Runtime > Run all
4. Espera: Todas las cells ejecutan sin errores

## 🐛 Si algo falla

### Local

```bash
# Reinstalar paquete
pip uninstall intention-collapse -y
pip install -e . --force-reinstall

# Verificar de nuevo
python tests/test_package_structure.py
```

### Google Colab

```python
# Limpiar y reinstalar
!rm -rf intention-collapse-experiments
!git clone -b v2-router-experiments https://github.com/patriciomvera/intention-collapse-experiments.git
!pip install -e /content/intention-collapse-experiments/ --force-reinstall

# Restart runtime
# Runtime > Restart runtime

# Verificar de nuevo
!python /content/intention-collapse-experiments/verify_colab_setup.py
```

## 📋 Status Indicators

- `[OK]` - Test pasó correctamente
- `[SKIP]` - Test omitido (esperado si faltan dependencias)
- `[FAIL]` - Test falló (requiere atención)
- `[SUCCESS]` - Todos los tests pasaron

## 🔗 Referencias Rápidas

- **Setup completo:** [COLAB_INSTALL.md](COLAB_INSTALL.md)
- **Resumen de cambios:** [PACKAGE_FIX_SUMMARY.md](PACKAGE_FIX_SUMMARY.md)
- **Tests:** [tests/README.md](tests/README.md)
