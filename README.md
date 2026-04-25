# Product Matching — Recopilación y Preparación de Datos

**Materia:** Ciencia de Datos Aplicada — ITBA
**Entregable:** 2 — *Recopilación y preparación de datos*

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
│   ├── 01_dataset_description.ipynb           # Notebook fuente
│   └── 01_dataset_description_executed.ipynb  # Notebook ejecutado (con outputs)
├── docs/
│   ├── [02] Recopilación y preparación de datos.pdf
│   └── Propuesta de Proyecto_ Product Matching.pptx
├── gen_notebook.py                 # Genera el notebook programáticamente
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

## Cómo reproducir

### Requisitos
- Python ≥ 3.10
- Paquetes: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `nbformat`, `nbconvert`, `ipykernel`

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

---

## Fuentes

- Papadakis, G. et al. (2022). *Datasets for Supervised Matching in Clean-Clean Entity Resolution*. Zenodo. <https://zenodo.org/records/7252010>
- Magellan project, Carnegie Mellon University.
