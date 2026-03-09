# ✅ TAREA 3 COMPLETADA: Option-Normalized Entropy

## 🎯 Objetivo

Implementar **option-normalized entropy** H_opt(I) para preguntas de múltiple choice (ARC, AQUA), que mide la entropía solo sobre tokens válidos [A,B,C,D,E] en lugar de todo el vocabulario.

## 🔑 Problema Resuelto

La entropía estándar H_int(I) confunde dos problemas diferentes en preguntas MC:

1. **COMPETENCE**: ¿Sabe el modelo la respuesta correcta?
2. **COMPLIANCE**: ¿Formateará el modelo la respuesta correctamente?

**Option-normalized entropy** separa estos problemas midiendo solo la incertidumbre sobre las opciones válidas.

## 📁 Archivos Creados/Modificados

### Nuevas Funciones en `src/metrics.py`

```python
✅ get_option_token_ids(tokenizer, options=['A','B','C','D','E'])
   - Obtiene token IDs para opciones MC
   - Maneja diferentes esquemas de tokenización ('A', ' A', 'A.', etc.)

✅ compute_option_normalized_entropy(logits, tokenizer, options, return_probs=False)
   - Calcula H_opt(I) solo sobre opciones válidas
   - Retorna entropy en bits (y opcionalmente distribución de probabilidades)

✅ compute_entropy_decomposition(logits, tokenizer, options, top_k=100)
   - Descompone entropy en componentes: standard vs option-normalized
   - Retorna ratio, probability mass, most likely option, confidence

✅ IntentionMetrics dataclass actualizado
   - Añadidos campos: option_normalized_entropy, option_probs, option_probability_mass
```

### Integración en `src/router/adaptive_router.py`

```python
✅ Constructor actualizado
   - Parámetro: use_option_normalized_entropy (auto-detecta para 'arc'/'aqua')
   - Define opciones válidas por benchmark (4 para ARC, 5 para AQUA)

✅ compute_intention_entropy() actualizado
   - Usa option-normalized entropy automáticamente para MC benchmarks
   - Parámetro return_decomposition para análisis detallado

✅ Thresholds ajustados para MC
   - GSM8K: low=0.5, high=1.2 (estándar)
   - ARC:   low=0.8, high=1.5 (ajustado para max_entropy=2.0 bits)
   - AQUA:  low=0.9, high=1.6 (ajustado para max_entropy=2.32 bits)
```

### Documentación y Ejemplos

```
✅ docs/option_normalized_entropy.md (documento completo)
   - Motivación matemática
   - Definiciones y bounds teóricos
   - Ejemplos de uso
   - Interpretación de métricas
   - API reference

✅ examples/option_normalized_demo.py (4 demos interactivos)
   - Demo 1: Comparación entropy estándar vs option-normalized
   - Demo 2: Preguntas fáciles vs difíciles
   - Demo 3: Impacto en decisiones de routing
   - Demo 4: Cómo se encodean los tokens de opciones

✅ tests/test_option_normalized_entropy.py (6 tests unitarios)
   - Test distribución uniforme (4 opciones → 2.0 bits)
   - Test distribución cierta (p=1 → 0 bits)
   - Test probabilidades suman a 1
   - Test descomposición de entropy
   - Test extracción de token IDs
   - Test 5 opciones AQUA (→ 2.32 bits)
```

## 🔬 Funcionalidad Clave

### 1. Cálculo de Option-Normalized Entropy

```python
from metrics import compute_option_normalized_entropy

# Obtener logits del modelo
inputs = tokenizer(prompt, return_tensors="pt")
logits = model(**inputs).logits[0, -1, :]

# Calcular entropy solo sobre [A,B,C,D]
entropy = compute_option_normalized_entropy(
    logits,
    tokenizer,
    options=['A', 'B', 'C', 'D']
)

print(f"Option-normalized entropy: {entropy:.3f} bits")
# Output: "Option-normalized entropy: 1.234 bits"
# (vs standard entropy que podría ser 3-5 bits)
```

### 2. Descomposición de Entropy

```python
from metrics import compute_entropy_decomposition

decomp = compute_entropy_decomposition(
    logits, tokenizer, options=['A','B','C','D']
)

print(f"Standard:        {decomp['standard']:.3f} bits")
print(f"Option-norm:     {decomp['option_normalized']:.3f} bits")
print(f"Ratio:           {decomp['ratio']:.3f}")
print(f"Prob mass:       {decomp['option_probability_mass']:.1%}")
print(f"Most likely:     {decomp['most_likely_option']} ({decomp['confidence']:.1%})")

# Interpretación del ratio:
# ratio > 0.8  → Uncertainty sobre qué opción (COMPETENCE issue)
# ratio < 0.3  → Uncertainty fuera de opciones (COMPLIANCE issue)
# 0.3-0.8      → Mixed uncertainty
```

### 3. Integración con Router

```python
from router import AdaptiveInferenceRouter

# Router para preguntas MC con option-normalized entropy
router = AdaptiveInferenceRouter(
    model=model,
    tokenizer=tokenizer,
    benchmark='arc',
    use_option_normalized_entropy=True,  # Auto-detectado para 'arc'/'aqua'
    entropy_threshold_low=0.8,           # Ajustado para MC
    entropy_threshold_high=1.5,
    verbose=True
)

result = router.generate(
    question="Which property can be measured with a ruler?",
    choices="A. mass\nB. temperature\nC. length\nD. volume",
    ground_truth="C"
)

# El router usa H_opt en lugar de H_int para routing
print(f"Option-normalized entropy: {result.intention_entropy:.3f} bits")
print(f"Route: {result.route_taken}")  # 'direct' o 'cot'
```

## 📊 Métricas y Bounds Teóricos

| Número Opciones | Max Entropy | Interpretación |
|-----------------|-------------|----------------|
| 2 opciones      | 1.00 bits   | Binary choice |
| 3 opciones      | 1.58 bits   | —— |
| 4 opciones (ARC)| 2.00 bits   | Uniforme = 25% cada opción |
| 5 opciones (AQUA)| 2.32 bits  | Uniforme = 20% cada opción |

**Comparación con Standard Entropy:**
- Standard H_int: 5-8 bits típicamente (vocabulario de 50k tokens)
- Option H_opt: 0-2.32 bits (solo 4-5 opciones)
- **Ratio H_opt/H_int revela si uncertainty es sobre respuesta o formato**

## 🎓 Casos de Uso

### Caso 1: Model Sabe la Respuesta (Competence OK)

```
Question: "What is 2 + 2?"
Options: A. 3  B. 4  C. 5  D. 6

Standard entropy:    4.1 bits  ← Alto (mucha uncertainty en vocab)
Option entropy:      0.3 bits  ← Bajo (sabe que es B)
Ratio:               0.07      ← Muy bajo
Prob mass on opts:   0.45
Most likely:         B (90% de masa en opciones)

Diagnóstico: COMPLIANCE issue (sabe respuesta pero formato incierto)
Acción:      Route to direct + usar constrained decoding
```

### Caso 2: Model No Sabe la Respuesta (Competence Issue)

```
Question: "What is the capital of Bhutan?"
Options: A. Thimphu  B. Kathmandu  C. Dhaka  D. Colombo

Standard entropy:    3.2 bits
Option entropy:      1.9 bits  ← Alto (casi uniforme)
Ratio:               0.59      ← Medio-alto
Prob mass on opts:   0.85
Option probs:        A:35% B:22% C:20% D:23%

Diagnóstico: COMPETENCE issue (no sabe la respuesta)
Acción:      Route to CoT para razonamiento
```

## 🔧 Thresholds Recomendados

| Benchmark | Opciones | Max Entropy | Low Threshold | High Threshold |
|-----------|----------|-------------|---------------|----------------|
| GSM8K     | N/A      | N/A         | 0.5           | 1.2            |
| ARC       | 4        | 2.0 bits    | 0.8           | 1.5            |
| AQUA      | 5        | 2.32 bits   | 0.9           | 1.6            |

**Regla general:**
- `low_threshold ≈ 0.35 × max_entropy`
- `high_threshold ≈ 0.65 × max_entropy`

## 🧪 Validación

### Hipótesis

> "Para preguntas MC donde el modelo sabe la respuesta,
>  H_opt debe ser BAJO incluso si H_int es ALTO."

### Experimento (a realizar en Tarea 5)

1. Evaluar 200 problemas ARC
2. Para cada problema:
   - Calcular H_int (standard)
   - Calcular H_opt (option-normalized)
   - Registrar si el modelo acertó
3. Comparar correlaciones:
   - H_int vs correctness
   - H_opt vs correctness

**Resultado esperado:**
- H_opt tiene **mayor correlación** con correctness
- H_opt es **mejor predictor** para routing

## 🔗 Conexión con Otras Tareas

### Tarea 4: Constrained Decoding
```python
# Medir intención con option-normalized entropy
if entropy < 0.8:  # Low uncertainty
    # Pero forzar formato válido por si acaso
    output = constrained_mc_generation(model, prompt, valid_options=['A','B','C','D'])
```

### Tarea 5: Router Experiment Notebook
```python
# Comparar routers:
# 1. Standard entropy router
# 2. Option-normalized entropy router
# Métrica: Accuracy + Token efficiency
```

### Tarea 6: Self-Consistency
```python
if option_entropy > 1.5:  # High uncertainty sobre respuesta
    # Generar múltiples respuestas, tomar majority vote
    answer = self_consistency_baseline(model, prompt, n_samples=5)
```

## 📈 Beneficios Esperados

1. **Mejor señal para routing**: H_opt correlaciona más con correctness
2. **Diagnóstico**: Separar competence de compliance
3. **Thresholds más interpretables**: Max entropy conocido (2.0 o 2.32 bits)
4. **Reducción de falsos positivos**: No confundir formato con conocimiento

## 🚀 Uso Rápido

```python
# Importar
from router import AdaptiveInferenceRouter

# Crear router (auto-detecta option-normalized para 'arc')
router = AdaptiveInferenceRouter(
    model=model,
    tokenizer=tokenizer,
    benchmark='arc'  # Automáticamente usa option-normalized entropy
)

# Generar
result = router.generate(
    question="Your MC question here",
    choices="A. ...\nB. ...\nC. ...\nD. ...",
    ground_truth="C"
)

# Revisar
print(f"Entropy: {result.intention_entropy:.3f} bits")  # Option-normalized
print(f"Route: {result.route_taken}")
print(f"Answer: {result.extracted_answer}")
print(f"Correct: {result.is_correct}")
```

## 📚 Documentación Completa

- **Teoría**: `docs/option_normalized_entropy.md`
- **Demos**: `examples/option_normalized_demo.py` (ejecutar para ver ejemplos)
- **Tests**: `tests/test_option_normalized_entropy.py`
- **API**: Ver docstrings en `src/metrics.py`

## ✅ Checklist de Completitud

- [x] Implementar `get_option_token_ids()`
- [x] Implementar `compute_option_normalized_entropy()`
- [x] Implementar `compute_entropy_decomposition()`
- [x] Actualizar dataclass `IntentionMetrics`
- [x] Integrar en `AdaptiveInferenceRouter`
- [x] Auto-detección para benchmarks MC
- [x] Ajustar thresholds para MC
- [x] Crear demos interactivos (4 demos)
- [x] Crear tests unitarios (6 tests)
- [x] Documentación completa
- [x] Verificar compilación de código

## 🎯 Próximos Pasos

**Tarea 4: Constrained Decoding**
- Forzar al modelo a solo emitir tokens válidos [A,B,C,D]
- Resolver compliance issues completamente
- Combinar con option-normalized entropy

O continuar con:
- **Tarea 5**: Notebook de experimento completo (200 problemas GSM8K)
- **Tarea 6**: Self-consistency control (reemplazar babble baseline)

---

**Fecha de completitud**: 2026-02-04
**Archivos afectados**: 6 nuevos, 2 modificados
**Líneas de código**: ~600 líneas (implementación + tests + demos)
