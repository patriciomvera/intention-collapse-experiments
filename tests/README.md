# Testing Guide

Este directorio contiene tests para verificar la instalación y estructura del paquete.

## Tests Disponibles

### 1. `test_package_structure.py` (Sin dependencias)

Test rápido que verifica la estructura del paquete sin requerir dependencias pesadas (torch, transformers).

**Uso:**
```bash
python tests/test_package_structure.py
```

**Verifica:**
- ✅ El paquete `src` se puede importar
- ✅ Todos los `__init__.py` existen en los lugares correctos
- ✅ Los módulos Python (.py) están presentes
- ✅ El mecanismo de lazy imports funciona
- ✅ `__all__` está definido correctamente

**Ideal para:** Verificar que la estructura del paquete es correcta antes de instalar dependencias.

### 2. `test_imports.py` (Requiere dependencias completas)

Test completo que verifica que todos los imports funcionen correctamente.

**Uso:**
```bash
# Primero instalar el paquete con dependencias
pip install -e .

# Luego ejecutar el test
python tests/test_imports.py
```

**Verifica:**
- ✅ `src.router` imports (AdaptiveInferenceRouter, RouteDecision, RouterResult)
- ✅ `src.metrics` imports (compute_intention_entropy, IntentionMetrics)
- ✅ `src.controls` imports (self_consistency_baseline, SelfConsistencyResult)
- ✅ `src.decoding` imports (constrained_mc_generation, MultipleChoiceLogitsProcessor)
- ✅ Funcionalidad básica (enums, clases)

**Ideal para:** Verificación completa después de instalar en Google Colab.

## Google Colab

Para verificar la instalación en Google Colab, usa el script dedicado en la raíz:

```python
!python /content/intention-collapse-experiments/verify_colab_setup.py
```

Este script ejecuta tests completos incluyendo:
- Versión de Python
- Instalación del paquete
- Dependencias
- Imports críticos
- Funcionalidad básica

## CI/CD

Para integración continua, ejecuta ambos tests:

```bash
# Test 1: Estructura (siempre debe pasar)
python tests/test_package_structure.py

# Test 2: Imports completos (requiere dependencias instaladas)
pip install -e .
python tests/test_imports.py
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"

**Problema:** Las dependencias no están instaladas.

**Solución:**
```bash
pip install -e .
```

### "ModuleNotFoundError: No module named 'src'"

**Problema:** El paquete no está instalado.

**Solución:**
```bash
# Local
pip install -e .

# Google Colab
!pip install -e /content/intention-collapse-experiments/
```

### Tests pasan pero los imports fallan en notebooks

**Problema:** Path incorrecto o instalación no detectada.

**Solución en Colab:**
```python
# Reinstalar paquete
!pip install -e /content/intention-collapse-experiments/ --force-reinstall

# Verificar
!python /content/intention-collapse-experiments/verify_colab_setup.py
```

## Estructura de Tests

```
tests/
├── README.md                     # Este archivo
├── test_package_structure.py     # Test sin dependencias
└── test_imports.py                # Test con dependencias completas
```

## Añadir Nuevos Tests

Para añadir un nuevo test:

1. Crear archivo `test_*.py` en este directorio
2. Usar formato `[OK]` / `[FAIL]` para mensajes
3. Return `True` si pasa, `False` si falla
4. Documentar en este README

## Referencias

- **Instalación Colab:** Ver [COLAB_INSTALL.md](../COLAB_INSTALL.md)
- **Documentación general:** Ver [README.md](../README.md)
