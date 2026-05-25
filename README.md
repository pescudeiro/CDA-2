# Product Matching — Ciencia de Datos Aplicada

**Materia:** Ciencia de Datos Aplicada — ITBA
**Entregables:** 2 — *Recopilación y preparación de datos* · 3 — *Modelado de la solución*

---

## Descripción

Este repositorio contiene la adquisición, EDA y preparación de datos para el proyecto **Product Matching**: dado un mismo producto descrito por dos vendedores distintos (e.g. Amazon vs Google), determinar si se trata del mismo ítem o no.

Trabajamos con **3 datasets** del benchmark de [Papadakis et al. (2022) en Zenodo](https://zenodo.org/records/7252010), elegidos por dificultad progresiva:

| Dataset | Table A | Table B | Pares | Características |
|---------|---------|---------|-------|-----------------|
| **Amazon-Google** | 1.363 | 3.226 | ~11.500 | Estructurado, software |
| **Walmart-Amazon** | 2.554 | 22.074 | ~10.200 | Estructurado, mayor escala |
| **Abt-Buy** | 1.081 | 1.092 | ~9.600 | Textual puro |

---

## Estructura del repositorio

```
.
├── README.md                       # Este archivo
├── PRESENTACION.md                 # Guion de la presentación oral
├── data/
│   ├── existingDatasets/           # Datos crudos (3 datasets, train/valid/test)
│   ├── label_distribution.png      # Match vs No-Match
│   ├── missing_values.png          # Faltantes por columna
│   ├── price_distribution.png      # Histograma de precios
│   ├── price_outliers.png          # Boxplot precios (escala log)
│   ├── price_standardized.png      # Z-score de precios
│   ├── text_length_distribution.png # Longitud de texto A vs B
│   └── feature_correlation.png     # Correlación features ↔ label
├── notebooks/
│   ├── 01_dataset_description.ipynb           # Entrega 2 — EDA + preparación (fuente)
│   ├── 01_dataset_description_executed.ipynb  # Entrega 2 — ejecutado (con outputs)
│   ├── 02_modeling.ipynb                       # Entrega 3 — modelado (fuente)
│   └── 02_modeling_executed.ipynb             # Entrega 3 — ejecutado (con outputs)
├── models/                          # Entrega 3 — modelo persistido
│   ├── product_matcher.joblib       # Bundle reutilizable (clasificador + TF-IDF + umbral + ...)
│   └── metrics.json                 # Métricas de test y ablation
├── docs/
│   ├── [02] Recopilación y preparación de datos.pdf
│   ├── [03] Modelado de la solución.pdf
│   └── Propuesta de Proyecto_ Product Matching.pptx
├── gen_notebook.py                 # Genera el notebook de la entrega 2
├── gen_modeling_notebook.py        # Genera el notebook de la entrega 3
└── run_notebook.py                 # Ejecuta el notebook end-to-end
```

---

## Contenido del notebook

El notebook `01_dataset_description_executed.ipynb` cubre los 5 bloques del enunciado:

1. **Descripción de los datasets** — origen (Zenodo), formato (CSV), variables y justificación.
2. **EDA** — tipos de variables, distribuciones, estadísticas descriptivas (incluyendo skewness y kurtosis), análisis de texto, distribución de labels, ejemplos de match / no-match / falsos positivos potenciales.
3. **Diagnóstico de calidad** — faltantes, duplicados, outliers (regla 1.5×IQR por tabla), inconsistencias (IDs orfanos en los splits).
4. **Transformaciones** — normalización de texto (lowercase + sin puntuación), winsorización de outliers, label encoding unificado A+B, generación de features (`word_overlap`, `brand_match`, `category_match`, `price_diff`, `price_ratio`, `has_both_prices`, `len_diff`), filtrado de variables.
5. **Reflexión final** — decisiones tomadas, dificultades y próximos pasos.

---

## Hallazgos principales del EDA

- **Desbalance de clases**: ~9–13 % de matches en los tres datasets → exige métricas tipo F1 / PR-AUC.
- **Distribución de precios fuertemente sesgada**: skewness entre 2,7 y 25,5 → winsorización p99 justificada.
- **Heterogeneidad léxica entre fuentes**: longitudes de título distintas entre Table A y B (Walmart-Amazon es el caso extremo).
- **Pocas marcas compartidas**: en Walmart-Amazon, sólo 400 de 1.972 marcas aparecen en ambas tablas → el matching exacto de marca no alcanza.
- **`word_overlap` (Jaccard) es el feature más discriminativo**: correlación con `label` de **+0,43 / +0,37 / +0,44** en los tres datasets respectivamente. Valida el approach textual.
- **`category_match` no aporta señal** en Walmart-Amazon (–0,01) — hallazgo contraintuitivo, probablemente por taxonomías inconsistentes entre fuentes.

---

## Modelado de la solución (Entregable 3)

> Notebook: `notebooks/02_modeling_executed.ipynb` · **Solo se usa el dataset Walmart-Amazon** (los otros fueron descartados por la cátedra).

### Enfoque
El problema es **product matching / entity resolution** planteado como **clasificación binaria supervisada de pares candidatos**: dado un par (registro de Walmart, registro de Amazon) predecir si son el mismo producto (`match`). Los splits etiquetados ya vienen provistos, lo que justifica el enfoque supervisado frente a clustering o RAG.

### Features (espacio híbrido)
- **Estructurales** (del entregable 2): `word_overlap` (Jaccard), `brand_match`, `category_match`, `price_diff`, `price_ratio`, `has_both_prices`, `len_title_diff`, más `modelno_match` (agregado aquí).
- **Texto clásico:** `tfidf_cos` — coseno TF-IDF (1-2 gramas) entre títulos.
- **Transformer (clase C10):** `emb_cos` — coseno de embeddings `sentence-transformers` (`all-MiniLM-L6-v2`); captura similitud semántica que TF-IDF no detecta.

### Modelos y validación
Tres modelos de complejidad creciente — **Regresión Logística** (baseline), **Random Forest** y **XGBoost** — con manejo de desbalance (`class_weight='balanced'` / `scale_pos_weight`). El **umbral de decisión se ajusta en validación** (máximo F1) y se reporta en test. Métricas apropiadas al desbalance (~9.4% positivos): **F1, PR-AUC, ROC-AUC**, precisión/recall y matriz de confusión.

### Cómputo en GPU (CUDA)
La generación de embeddings del transformer y el entrenamiento de XGBoost corren en **GPU NVIDIA (CUDA)** cuando está disponible (probado en RTX 3090); el notebook detecta el dispositivo y cae a CPU si no hay GPU. Regresión Logística y Random Forest se entrenan en CPU (scikit-learn no soporta GPU). El bundle persiste XGBoost en modo CPU para que se recargue sin GPU.

### Resultados (test)

| Modelo | F1 | Precisión | Recall | PR-AUC | ROC-AUC |
|--------|----|-----------|--------|--------|---------|
| Regresión Logística | 0.709 | 0.731 | 0.689 | 0.799 | 0.938 |
| **Random Forest** (mejor) | **0.760** | 0.901 | 0.658 | 0.816 | 0.951 |
| XGBoost | 0.747 | 0.864 | 0.658 | 0.803 | 0.943 |

**Ablation (XGBoost)** — cada bloque de señal aporta valor:

| Features | F1 | PR-AUC | ROC-AUC |
|----------|----|--------|---------|
| Estructural | 0.662 | 0.663 | 0.848 |
| + TF-IDF | 0.751 | 0.795 | 0.936 |
| + Transformer | 0.752 | 0.803 | 0.944 |

La similitud de título (`emb_cos`/`tfidf_cos`/`word_overlap`) domina la importancia de features; el transformer suma señal semántica incremental sobre el texto clásico.

### Persistencia
El mejor modelo se guarda en `models/product_matcher.joblib` como **bundle auto-contenido** (clasificador + scaler + vectorizador TF-IDF + parámetros de limpieza + orden de features + umbral + id del modelo de embeddings). El notebook incluye una celda que **recarga el bundle y reproduce F1=0.760 exacto sin reentrenar**, dejando la solución lista para integrarse en el 4.º entregable.

---

## Cómo reproducir

### Requisitos
- Python ≥ 3.10
- Entrega 2: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `nbformat`, `nbconvert`, `ipykernel`
- Entrega 3 (además): `xgboost`, `torch`, `sentence-transformers`, `joblib`
  - Para GPU: instalar `torch` con CUDA, p. ej. `pip install torch --index-url https://download.pytorch.org/whl/cu128`

### Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib seaborn scikit-learn scipy nbformat nbconvert ipykernel
python -m ipykernel install --user --name cda-venv --display-name "CDA venv"
```

### Ejecución
```bash
# Regenerar el notebook desde la fuente programática
python gen_notebook.py

# Ejecutar el notebook end-to-end (escribe outputs a 01_dataset_description_executed.ipynb)
python run_notebook.py
```

Alternativamente, abrir `notebooks/01_dataset_description_executed.ipynb` en Jupyter y ejecutar Run All.

Para la entrega 3 (requiere haber corrido antes la entrega 2, que genera `data/all_features.pkl`):

```bash
# Regenerar el notebook de modelado
python gen_modeling_notebook.py

# Ejecutar end-to-end (usa GPU/CUDA si está disponible; descarga el transformer la primera vez)
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=cda-venv \
  --output 02_modeling_executed.ipynb notebooks/02_modeling.ipynb
```

---

## Fuentes

- Papadakis, G. et al. (2022). *Datasets for Supervised Matching in Clean-Clean Entity Resolution*. Zenodo. <https://zenodo.org/records/7252010>
- Magellan project, Carnegie Mellon University.
