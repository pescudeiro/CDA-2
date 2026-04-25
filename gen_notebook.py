#!/usr/bin/env python3
"""Generate the complete Product Matching notebook programmatically."""
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
cells = []

# Helper to add markdown
def md(text):
    cells.append(new_markdown_cell(text))

# Helper to add code
def code(text, name=None):
    cells.append(new_code_cell(text))

# ============ TITLE ============
md("""# Product Matching - Recopilación y Preparación de Datos

**Integrantes:** Patricio Escudeiro (61156), Jose Burgos (61525)

**Entregable:** Segundo entregable - Ciencia de Datos Aplicada

---

## 1. Descripción de los Datasets

### 1.1 Origen y Formato

**Fuente principal:** Zenodo - *Datasets for Supervised Matching in Clean-Clean Entity Resolution*
**Autores:** George Papadakis et al. (2022)
**URL:** https://zenodo.org/records/7252010
**Licencia:** Open Access

El repositorio incluye datasets consolidados del proyecto Magellan (CMU) más nuevos benchmarks generados con DeepBlocker. Utilizamos tres datasets estructurados de Record Linkage:

### 1.2 Datasets Seleccionados

| Dataset | Table A | Table B | Train | Valid | Test | Total Pares | Tamaño |
|---------|---------|---------|-------|-------|------|-------------|--------|
| **Amazon-Google Products** | 1,363 | 3,226 | 6,874 | 2,293 | 2,293 | ~11,500 | 1.8 MB |
| **Walmart-Amazon** | 2,554 | 22,074 | 6,144 | 2,049 | 2,049 | ~10,200 | 4.9 MB |
| **Abt-Buy (Textual)** | 1,081 | 1,092 | 5,743 | 1,916 | 1,916 | ~9,600 | 4.5 MB |

**Tamaño total de datos extraídos:** 12 MB
**Archivo comprimido original:** 587 MB (incluye 13 datasets)

### 1.3 Variables por Dataset

---

**Amazon-Google Products (Structured)**

| Campo | Descripción |
|-------|-------------|
| `id` | Identificador único del producto |
| `title` | Título del producto (campo principal para matching) |
| `manufacturer` | Marca/fabricante del producto |
| `price` | Precio del producto |

---

**Walmart-Amazon (Structured)**

| Campo | Descripción |
|-------|-------------|
| `id` | Identificador único del producto |
| `title` | Título del producto |
| `category` | Categoría del producto |
| `brand` | Marca del producto |
| `modelno` | Número de modelo |
| `price` | Precio del producto |

---

**Abt-Buy (Textual)**

| Campo | Descripción |
|-------|-------------|
| `id` | Identificador único del producto |
| `name` | Nombre del producto |
| `description` | Descripción textual del producto |
| `price` | Precio del producto |

---

**Archivos de Train/Valid/Test (común a todos los datasets)**

| Campo | Descripción |
|-------|-------------|
| `_id` | Identificador del par |
| `label` | **0 = No Match, 1 = Match** (ground truth) |
| `table1.id` | ID del producto en Table A |
| `table2.id` | ID del producto en Table B |
| `table1.<field>` | Campo correspondiente de Table A |
| `table2.<field>` | Campo correspondiente de Table B |

### 1.4 Justificación de la Elección

Estos tres datasets son ideales para nuestro proyecto de **Product Matching** porque:

1. **Representan el problema real de Record Linkage:** Mismo producto, múltiples descripciones entre fuentes heterogéneas
2. **Tienen ground truth (Perfect Mapping):** Labels de matching para entrenar y evaluar modelos supervisados
3. **Dificultad progresiva:**
   - Amazon-Google: Matching estructurado con campos claros (título, marca, precio)
   - Walmart-Amazon: Mayor escala, más campos estructurados (categoría, modelo)
   - Abt-Buy: Matching textual puro - requiere NLP para extraer señales de texto descriptivo
4. **Ruido extremo:** Títulos con formatos completamente distintos entre fuentes
5. **Falsos positivos potenciales:** Productos similares pero distintos (ej: misma versión en diferente idioma, modelos similares de misma marca)
6. **Escalabilidad:** Permiten evaluar el modelo en diferentes dominios de productos
""")

# ============ IMPORTS ============
md("## 2. Carga de Datos")
code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# Configurar estilos
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# Ruta base de los datos
DATA_DIR = '../data/existingDatasets'

print("Librerías cargadas correctamente.")
print(f"Directorio de datos: {DATA_DIR}")""")

# ============ LOAD DATASET ============
code('''def load_dataset(name):
    """Cargar un dataset completo (tables + splits)"""
    base = os.path.join(DATA_DIR, name)
    
    # Cargar tablas
    tableA = pd.read_csv(os.path.join(base, 'tableA.csv'), dtype=str)
    tableB = pd.read_csv(os.path.join(base, 'tableB.csv'), dtype=str)
    
    # Cargar splits
    train = pd.read_csv(os.path.join(base, 'train.csv'), dtype=str)
    valid = pd.read_csv(os.path.join(base, 'valid.csv'), dtype=str)
    test = pd.read_csv(os.path.join(base, 'test.csv'), dtype=str)
    
    # Convertir label a int
    for df in [train, valid, test]:
        df['label'] = df['label'].astype(int)
    
    # Convertir precios a float (NaN si vacío)
    for table in [tableA, tableB]:
        if 'price' in table.columns:
            table['price'] = pd.to_numeric(table['price'], errors='coerce')
    
    return {
        'tableA': tableA,
        'tableB': tableB,
        'train': train,
        'valid': valid,
        'test': test
    }

# Cargar los tres datasets
datasets = {
    'amazon_google': load_dataset('structured_amazon_google'),
    'walmart_amazon': load_dataset('structured_walmart_amazon'),
    'abt_buy': load_dataset('textual_abt_buy')
}

print("Datasets cargados:")
for name, ds in datasets.items():
    print(f"\\n  === {name} ===")
    print(f"  Table A: {len(ds['tableA'])} registros")
    print(f"  Table B: {len(ds['tableB'])} registros")
    print(f"  Train: {len(ds['train'])} pares")
    print(f"  Valid: {len(ds['valid'])} pares")
    print(f"  Test: {len(ds['test'])} pares")''')

# ============ EDA ============
md("## 3. Análisis Exploratorio de Datos (EDA)")
md("### 3.1 Distribución de Labels (Match vs No Match)")
code('''fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, ds) in zip(axes, datasets.items()):
    all_labels = pd.concat([ds['train']['label'], ds['valid']['label'], ds['test']['label']])
    counts = all_labels.value_counts().sort_index()
    
    bars = ax.bar(['No Match (0)', 'Match (1)'], counts.values, 
                   color=['#e74c3c', '#2ecc71'], alpha=0.8)
    
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
                f'{val}\\n({val/counts.sum()*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_title(name.replace('_', '-').title(), fontsize=13, fontweight='bold')
    ax.set_ylabel('Cantidad de pares')
    ax.set_ylim(0, max(counts.values) * 1.15)

plt.suptitle('Distribución de Labels: Match vs No Match', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../data/label_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# Imprimir ratios
print("\\nRatios Match/No Match:")
for name, ds in datasets.items():
    all_labels = pd.concat([ds['train']['label'], ds['valid']['label'], ds['test']['label']])
    match_ratio = all_labels.mean()
    print(f"  {name}: {match_ratio:.2%} matches ({all_labels.sum()}/{len(all_labels)})")''')

# ============ VARIABLE TYPES ============
md("### 3.2 Tipos de Variables y Estadísticas Descriptivas")
code('''for name, ds in datasets.items():
    print(f"\\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")
    
    print("\\n--- Table A ---")
    print(f"Registros: {len(ds['tableA'])}")
    print(f"Columnas: {list(ds['tableA'].columns)}")
    print("\\nTipos de datos:")
    print(ds['tableA'].dtypes)
    print("\\nPrimeros 3 registros:")
    print(ds['tableA'].head(3).to_string())
    
    print("\\n--- Table B ---")
    print(f"Registros: {len(ds['tableB'])}")
    print(f"Columnas: {list(ds['tableB'].columns)}")
    
    # Estadísticas de precio
    if 'price' in ds['tableA'].columns:
        print("\\n--- Estadísticas de Precio (Table A) ---")
        price_a = pd.to_numeric(ds['tableA']['price'], errors='coerce')
        print(f"  Media: ${price_a.mean():.2f}")
        print(f"  Mediana: ${price_a.median():.2f}")
        print(f"  Min: ${price_a.min():.2f}")
        print(f"  Max: ${price_a.max():.2f}")
        print(f"  NaN: {price_a.isna().sum()} ({price_a.isna().mean()*100:.1f}%)")
    
    print()''')

# ============ TEXT ANALYSIS ============
md("### 3.3 Análisis de Campos de Texto")
code('''for name, ds in datasets.items():
    print(f"\\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")
    
    # Identificar campos de texto
    text_fields_a = [c for c in ds['tableA'].columns if c not in ['id', 'price']]
    
    for field in text_fields_a:
        values = ds['tableA'][field].dropna()
        if len(values) == 0:
            print(f"\\n  {field}: TODOS LOS VALORES SON NaN")
            continue
        
        # Longitud del texto
        lengths = values.astype(str).str.len()
        word_counts = values.astype(str).str.split().str.len()
        
        print(f"\\n  Campo: {field}")
        print(f"    No NaN: {len(values)}/{len(ds['tableA'])} ({len(values)/len(ds['tableA'])*100:.1f}%)")
        print(f"    Unique values: {values.nunique()}")
        print(f"    Longitud caracteres - Media: {lengths.mean():.1f}, Min: {lengths.min()}, Max: {lengths.max()}")
        print(f"    Longitud palabras - Media: {word_counts.mean():.1f}, Min: {word_counts.min()}, Max: {word_counts.max()}")
        
        # Ejemplos
        print(f"    Ejemplos:")
        for val in values.head(2).astype(str):
            print(f"      '{val[:100]}{'...' if len(val)>100 else ''}'")
    
    print()''')

# ============ TEXT LENGTH VISUALIZATION ============
md("### 3.4 Visualización: Distribución de Longitud de Texto")
code('''fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for idx, (name, ds) in enumerate(datasets.items()):
    # Campo principal de texto
    text_field = 'title' if 'title' in ds['tableA'].columns else 'name'
    
    # Distribución de longitud de caracteres
    lengths_a = ds['tableA'][text_field].dropna().astype(str).str.len()
    lengths_b = ds['tableB'][text_field].dropna().astype(str).str.len()
    
    # Histograma
    ax = axes[0, idx]
    ax.hist(lengths_a, bins=30, alpha=0.5, label=f'Table A (n={len(lengths_a)})', color='#3498db')
    ax.hist(lengths_b, bins=30, alpha=0.5, label=f'Table B (n={len(lengths_b)})', color='#e74c3c')
    ax.set_title(f'Longitud de {text_field} (caracteres)', fontweight='bold')
    ax.set_xlabel('Caracteres')
    ax.set_ylabel('Frecuencia')
    ax.legend()
    
    # Boxplot
    ax = axes[1, idx]
    ax.boxplot([lengths_a, lengths_b], labels=['Table A', 'Table B'])
    ax.set_title(f'Distribución de longitud - {name.replace("_", "-").title()}')
    ax.set_ylabel('Caracteres')

plt.suptitle('Análisis de Longitud de Texto por Dataset', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../data/text_length_distribution.png', dpi=150, bbox_inches='tight')
plt.show()''')

# ============ PRICE DISTRIBUTION ============
md("### 3.5 Distribución de Precios")
code('''fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, ds) in zip(axes, datasets.items()):
    price_a = pd.to_numeric(ds['tableA']['price'], errors='coerce')
    price_b = pd.to_numeric(ds['tableB']['price'], errors='coerce')
    
    # Filtrar valores razonables (< 10000 para evitar outliers extremos)
    price_a = price_a[(price_a > 0) & (price_a < 10000)]
    price_b = price_b[(price_b > 0) & (price_b < 10000)]
    
    ax.hist(price_a, bins=40, alpha=0.5, label='Table A', color='#3498db')
    ax.hist(price_b, bins=40, alpha=0.5, label='Table B', color='#e74c3c')
    ax.set_title(name.replace('_', '-').title(), fontweight='bold')
    ax.set_xlabel('Precio ($)')
    ax.set_ylabel('Frecuencia')
    ax.legend()
    ax.set_xlim(0, min(price_a.max(), price_b.max()) * 1.1)

plt.suptitle('Distribución de Precios por Dataset', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../data/price_distribution.png', dpi=150, bbox_inches='tight')
plt.show()''')

# ============ EXAMPLES ============
md("### 3.6 Ejemplos de Pares Match vs No Match")
code('''for name, ds in datasets.items():
    print(f"\\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")
    
    train = ds['train']
    
    # Identificar el campo de texto principal
    if 'table1.title' in train.columns:
        t1_col, t2_col = 'table1.title', 'table2.title'
    elif 'table1.name' in train.columns:
        t1_col, t2_col = 'table1.name', 'table2.name'
    else:
        t1_col, t2_col = 'table1.title', 'table2.title'
    
    # Ejemplos de MATCH
    matches = train[train['label'] == 1].head(3)
    print("\\n--- Ejemplos de MATCH (mismo producto) ---")
    for _, row in matches.iterrows():
        print(f"\\n  TableA: {row[t1_col][:120]}")
        print(f"  TableB: {row[t2_col][:120]}")
    
    # Ejemplos de NO MATCH
    no_matches = train[train['label'] == 0].head(3)
    print("\\n--- Ejemplos de NO MATCH (productos diferentes) ---")
    for _, row in no_matches.iterrows():
        print(f"\\n  TableA: {row[t1_col][:120]}")
        print(f"  TableB: {row[t2_col][:120]}")
    
    # Falsos positivos potenciales
    print("\\n--- Falsos Positivos Potenciales (productos similares) ---")
    if 'table1.manufacturer' in train.columns:
        t1_brand, t2_brand = 'table1.manufacturer', 'table2.manufacturer'
    elif 'table1.brand' in train.columns:
        t1_brand, t2_brand = 'table1.brand', 'table2.brand'
    else:
        t1_brand, t2_brand = None, None
    
    if t1_brand and t1_brand in train.columns and t2_brand in train.columns:
        similar = train[
            (train['label'] == 0) & 
            (train[t1_brand].notna()) & (train[t2_brand].notna()) &
            (train[t1_brand].str.lower() == train[t2_brand].str.lower())
        ].head(3)
        print(f"  Pares con misma marca pero NO match ({len(similar)} encontrados):")
        for _, row in similar.iterrows():
            print(f"\\n  [{row[t1_brand]}] TableA: {row[t1_col][:100]}")
            print(f"  [{row[t2_brand]}] TableB: {row[t2_col][:100]}")
    
    print()''')

# ============ DATA QUALITY ============
md("## 4. Diagnóstico y Calidad de Datos")
md("### 4.1 Datos Faltantes (Missing Values)")
code('''fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, (name, ds) in zip(axes, datasets.items()):
    # Combinar tables para análisis
    combined = pd.concat([ds['tableA'], ds['tableB']], ignore_index=True)
    
    # Calcular missing por columna
    missing = combined.isna().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=True)
    
    if len(missing) > 0:
        bars = ax.barh(missing.index, missing.values, color='#e74c3c', alpha=0.8)
        for bar, val in zip(bars, missing.values):
            ax.text(val + 0.5, bar.get_y() + bar.get_height()/2.,
                    f'{val:.1f}%', va='center', fontsize=10)
        ax.set_xlim(0, max(missing.values) * 1.15)
    else:
        ax.text(0.5, 0.5, 'Sin valores faltantes', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
    
    ax.set_title(name.replace('_', '-').title(), fontweight='bold')
    ax.set_xlabel('% de valores faltantes')

plt.suptitle('Valores Faltantes por Columna', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../data/missing_values.png', dpi=150, bbox_inches='tight')
plt.show()

# Imprimir resumen numérico
print("Resumen de valores faltantes:")
for name, ds in datasets.items():
    print(f"\\n  === {name} ===")
    for table_name in ['tableA', 'tableB']:
        table = ds[table_name]
        missing_counts = table.isna().sum()
        missing_pct = (missing_counts / len(table) * 100)
        print(f"  {table_name}:")
        for col in table.columns:
            if missing_counts[col] > 0:
                print(f"    {col}: {missing_counts[col]} ({missing_pct[col]:.1f}%)")
        if missing_counts.sum() == 0:
            print(f"    Sin valores faltantes")''')

# ============ DUPLICATES ============
md("### 4.2 Duplicados")
code('''print("Análisis de duplicados:")
for name, ds in datasets.items():
    print(f"\\n  === {name} ===")
    
    for table_name in ['tableA', 'tableB']:
        table = ds[table_name]
        
        # Duplicados por ID
        id_dups = table['id'].duplicated().sum()
        print(f"  {table_name}: {id_dups} IDs duplicados de {len(table)}")
        
        # Duplicados completos
        full_dups = table.duplicated().sum()
        print(f"  {table_name}: {full_dups} filas completamente duplicadas")
    
    # Duplicados en train/valid/test
    for split_name in ['train', 'valid', 'test']:
        split = ds[split_name]
        pair_dups = split.duplicated(subset=['table1.id', 'table2.id']).sum()
        print(f"  {split_name}: {pair_dups} pares duplicados")
    
    print()''')

# ============ OUTLIERS ============
md("### 4.3 Outliers en Precio")
code('''fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, ds) in zip(axes, datasets.items()):
    price_a = pd.to_numeric(ds['tableA']['price'], errors='coerce').dropna()
    price_b = pd.to_numeric(ds['tableB']['price'], errors='coerce').dropna()
    
    # Calcular IQR
    q1 = price_a.quantile(0.25)
    q3 = price_a.quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    
    outliers_a = (price_a > upper).sum()
    outliers_b = (price_b > upper).sum()
    
    bp = ax.boxplot([price_a, price_b], labels=['Table A', 'Table B'])
    ax.set_title(f'{name.replace("_", "-").title()}\\nOutliers: A={outliers_a}, B={outliers_b}', fontweight='bold')
    ax.set_ylabel('Precio ($)')
    ax.set_yscale('log')

plt.suptitle('Outliers en Precio (escala logarítmica)', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../data/price_outliers.png', dpi=150, bbox_inches='tight')
plt.show()

print("Resumen de outliers en precio:")
for name, ds in datasets.items():
    print(f"\\n  === {name} ===")
    for table_name in ['tableA', 'tableB']:
        price = pd.to_numeric(ds[table_name]['price'], errors='coerce').dropna()
        q1, q3 = price.quantile(0.25), price.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
        outliers = ((price < lower) | (price > upper)).sum()
        print(f"  {table_name}: Q1=${q1:.2f}, Q3=${q3:.2f}, IQR=${iqr:.2f}, Outliers: {outliers}/{len(price)}")''')

# ============ INCONSISTENCIES ============
md("### 4.4 Inconsistencias Detectadas")
code('''print("Inconsistencias detectadas:")
for name, ds in datasets.items():
    print(f"\\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")
    
    # Verificar que los IDs en train/valid/test existen en las tablas
    tableA_ids = set(ds['tableA']['id'].astype(str))
    tableB_ids = set(ds['tableB']['id'].astype(str))
    
    for split_name in ['train', 'valid', 'test']:
        split = ds[split_name]
        
        t1_found = split['table1.id'].astype(str).isin(tableA_ids).sum()
        t2_found = split['table2.id'].astype(str).isin(tableB_ids).sum()
        
        t1_orphan = len(split) - t1_found
        t2_orphan = len(split) - t2_found
        
        print(f"\\n  {split_name}:")
        print(f"    Table1 IDs encontrados: {t1_found}/{len(split)} ({t1_orphan} orfanos)")
        print(f"    Table2 IDs encontrados: {t2_found}/{len(split)} ({t2_orphan} orfanos)")
    
    # Verificar consistencia de labels
    all_labels = pd.concat([ds['train']['label'], ds['valid']['label'], ds['test']['label']])
    unique_labels = all_labels.unique()
    print(f"\\n  Labels únicos: {unique_labels}")
    if len(unique_labels) != 2:
        print(f"  ADVERTENCIA: Labels inesperados!")
    
    # Campos vacíos vs NaN
    for table_name in ['tableA', 'tableB']:
        table = ds[table_name]
        empty_strings = (table == '').sum()
        if empty_strings.sum() > 0:
            print(f"\\n  {table_name} - Campos con string vacío:")
            for col in table.columns:
                if empty_strings[col] > 0:
                    print(f"    {col}: {empty_strings[col]} strings vacíos")
    
    print()''')

# ============ TRANSFORMATIONS ============
md("## 5. Transformaciones Realizadas")
code('''def clean_dataset(name, ds):
    """Aplicar transformaciones de limpieza al dataset"""
    cleaned = {}
    
    for table_name in ['tableA', 'tableB']:
        table = ds[table_name].copy()
        
        # 1. Remover duplicados por ID
        table = table.drop_duplicates(subset=['id'], keep='first')
        
        # 2. Convertir strings vacíos a NaN
        table = table.replace('', np.nan)
        
        # 3. Strip whitespace en campos de texto
        text_cols = [c for c in table.columns if c not in ['id', 'price']]
        for col in text_cols:
            table[col] = table[col].astype(str).str.strip()
            table[col] = table[col].replace('nan', np.nan)
        
        # 4. Normalizar precio
        if 'price' in table.columns:
            table['price'] = pd.to_numeric(table['price'], errors='coerce')
            # Remover outliers extremos (> 99th percentile)
            price_upper = table['price'].quantile(0.99)
            table.loc[table['price'] > price_upper, 'price'] = np.nan
        
        cleaned[table_name] = table
    
    # 5. Procesar splits
    for split_name in ['train', 'valid', 'test']:
        split = ds[split_name].copy()
        split = split.replace('', np.nan)
        split['label'] = split['label'].astype(int)
        
        # Remover pares con IDs orfanos
        t1_ids = set(cleaned['tableA']['id'].astype(str))
        t2_ids = set(cleaned['tableB']['id'].astype(str))
        split = split[
            split['table1.id'].astype(str).isin(t1_ids) &
            split['table2.id'].astype(str).isin(t2_ids)
        ].reset_index(drop=True)
        
        cleaned[split_name] = split
    
    return cleaned

# Aplicar limpieza
cleaned_datasets = {}
for name, ds in datasets.items():
    cleaned_datasets[name] = clean_dataset(name, ds)
    
    print(f"\\n=== {name} - Transformaciones aplicadas ===")
    for table_name in ['tableA', 'tableB']:
        original = len(ds[table_name])
        cleaned = len(cleaned_datasets[name][table_name])
        print(f"  {table_name}: {original} -> {cleaned} registros ({original - cleaned} removidos)")
    
    for split_name in ['train', 'valid', 'test']:
        original = len(ds[split_name])
        cleaned = len(cleaned_datasets[name][split_name])
        print(f"  {split_name}: {original} -> {cleaned} pares ({original - cleaned} removidos)")''')

# ============ CATEGORICAL ENCODING ============
md("### 5.1 Codificación de Variables Categóricas")
code('''print("Codificación de variables categóricas:")
for name, ds in cleaned_datasets.items():
    print(f"\\n  === {name} ===")
    
    for table_name in ['tableA', 'tableB']:
        table = ds[table_name]
        
        # Identificar columnas categóricas
        cat_cols = [c for c in table.columns if c not in ['id', 'price', 'title', 'name', 'description']]
        
        for col in cat_cols:
            unique_vals = table[col].dropna().nunique()
            top_vals = table[col].value_counts().head(5)
            print(f"\\n  {table_name}.{col} ({unique_vals} valores únicos):")
            print(f"    Top 5: {dict(top_vals)}")''')

# ============ PRICE STANDARDIZATION ============
md("### 5.2 Normalización de Precio (Z-Score)")
code('''from sklearn.preprocessing import StandardScaler

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, ds) in zip(axes, cleaned_datasets.items()):
    price_a = ds['tableA']['price'].dropna()
    price_b = ds['tableB']['price'].dropna()
    
    # Estandarización
    scaler = StandardScaler()
    all_prices = pd.concat([price_a, price_b])
    scaler.fit(all_prices.values.reshape(-1, 1))
    
    price_a_scaled = scaler.transform(price_a.values.reshape(-1, 1))
    price_b_scaled = scaler.transform(price_b.values.reshape(-1, 1))
    
    ax.hist(price_a_scaled.flatten(), bins=30, alpha=0.5, label='Table A', color='#3498db')
    ax.hist(price_b_scaled.flatten(), bins=30, alpha=0.5, label='Table B', color='#e74c3c')
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.set_title(f'{name.replace("_", "-").title()}\\n(Estandarizado: mean=0, std=1)', fontweight='bold')
    ax.set_xlabel('Precio (Z-score)')
    ax.set_ylabel('Frecuencia')
    ax.legend()

plt.suptitle('Precio Estandarizado (Z-Score)', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../data/price_standardized.png', dpi=150, bbox_inches='tight')
plt.show()''')

# ============ FEATURE GENERATION ============
md("### 5.3 Generación de Nuevas Variables")
code('''def generate_features(ds, name):
    """Generar features para el matching"""
    features = {}
    
    for split_name in ['train', 'valid', 'test']:
        split = ds[split_name].copy()
        
        # Identificar campos disponibles
        has_title = 'table1.title' in split.columns
        has_name = 'table1.name' in split.columns
        has_manufacturer = 'table1.manufacturer' in split.columns
        has_brand = 'table1.brand' in split.columns
        has_price = 'table1.price' in split.columns
        
        # Feature 1: Longitud del texto
        if has_title:
            split['len_title_diff'] = (split['table1.title'].str.len() - split['table2.title'].str.len()).abs()
        elif has_name:
            split['len_name_diff'] = (split['table1.name'].str.len() - split['table2.name'].str.len()).abs()
        
        # Feature 2: Coincidencia de marca
        if has_manufacturer:
            split['brand_match'] = (
                split['table1.manufacturer'].str.lower().fillna('') == 
                split['table2.manufacturer'].str.lower().fillna('')
            ).astype(int)
        elif has_brand:
            split['brand_match'] = (
                split['table1.brand'].str.lower().fillna('') == 
                split['table2.brand'].str.lower().fillna('')
            ).astype(int)
        
        # Feature 3: Diferencia de precio
        if has_price:
            p1 = pd.to_numeric(split['table1.price'], errors='coerce')
            p2 = pd.to_numeric(split['table2.price'], errors='coerce')
            split['price_diff'] = (p1 - p2).abs()
            split['price_ratio'] = p1 / (p2 + 1e-8)
        
        # Feature 4: Overlap de palabras (Jaccard)
        if has_title:
            split['word_overlap'] = split.apply(
                lambda r: len(set(str(r['table1.title']).lower().split()) & 
                           set(str(r['table2.title']).lower().split())) / 
                           max(len(set(str(r['table1.title']).lower().split()) | 
                                  set(str(r['table2.title']).lower().split())), 1),
                axis=1
            )
        elif has_name:
            split['word_overlap'] = split.apply(
                lambda r: len(set(str(r['table1.name']).lower().split()) & 
                           set(str(r['table2.name']).lower().split())) / 
                           max(len(set(str(r['table1.name']).lower().split()) | 
                                  set(str(r['table2.name']).lower().split())), 1),
                axis=1
            )
        
        features[split_name] = split
    
    return features

# Generar features para cada dataset
all_features = {}
for name, ds in cleaned_datasets.items():
    all_features[name] = generate_features(ds, name)
    
    print(f"\\n=== {name} - Features generadas ===")
    train = all_features[name]['train']
    new_feature_cols = [c for c in train.columns if c not in datasets[name]['train'].columns]
    print(f"  Nuevas features: {new_feature_cols}")
    
    # Correlación de features con el label
    print("\\n  Correlación de features con label (match=1):")
    for col in new_feature_cols:
        if train[col].notna().any():
            corr = train[col].corr(train['label'])
            print(f"    {col}: {corr:.4f}")''')

# ============ REFLECTION ============
md("""## 6. Reflexión Final

### 6.1 Decisiones Tomadas y Justificación

**1. Selección de tres datasets complementarios:**
- *Amazon-Google:* Estructurado, campos claros, buen punto de partida
- *Walmart-Amazon:* Mayor escala, más campos estructurados, prueba de robustez
- *Abt-Buy:* Textual puro, prueba la capacidad del modelo NLP sin estructura

**2. Limpieza de datos:**
- Strings vacíos convertidos a NaN para tratamiento consistente
- Duplicados por ID eliminados (manteniendo el primero)
- Outliers de precio removidos (>99th percentile) para no sesgar las métricas
- Pares con IDs orfanos (que no existen en las tablas) eliminados

**3. Features generadas:**
- *Brand match:* Feature binaria potente - si la marca coincide, alta probabilidad de match
- *Price diff/ratio:* La diferencia de precio entre productos matched debería ser baja
- *Word overlap (Jaccard):* Similitud léxica básica como baseline
- *Length diff:* Los títulos de productos matched suelen tener longitudes similares

### 6.2 Dificultades Encontradas

1. **Desequilibrio extremo de clases:** ~90%+ no matches vs ~10% matches. Esto requiere:
   - Métricas apropiadas (F1-score, no accuracy)
   - Estrategias de balanceo (oversampling, class weights)
   - Evaluación cuidadosa para no tener un modelo que siempre prediga "no match"

2. **Falsos positivos potenciales:** Productos de la misma marca, categoría similar, pero distintos. El modelo debe distinguir entre:
   - "Microsoft Office 2007" vs "Microsoft Office 2007 [Spanish]" -> NO match
   - "Microsoft Office 2007" vs "Microsoft Office Home and Student 2007" -> NO match

3. **Ruido en los textos:** Títulos con diferentes formatos, mayúsculas, puntuación, y estructura entre fuentes.

4. **Valores faltantes en precio:** ~50-80% de los productos no tienen precio disponible.

### 6.3 Siguientes Pasos Proyectados

1. **Generación de Embeddings:** Usar sentence-transformers para crear representaciones vectoriales de los textos
2. **Modelos de matching:**
   - Baseline: Similitud de coseno entre embeddings
   - Modelo supervisado: Clasificador con features + embeddings
   - Modelo ensemble: Combinar señales estructurales y textuales
3. **Evaluación:** Matriz de confusión, F1-score, Precision-Recall curve
4. **Optimización:** Tuning de hiperparámetros, feature engineering adicional
5. **Deploy:** API para matching de productos en tiempo real
""")

# ============ SAVE ============
code('''# Guardar datasets limpios para el siguiente entregable
import pickle

with open('../data/cleaned_datasets.pkl', 'wb') as f:
    pickle.dump(cleaned_datasets, f)

with open('../data/all_features.pkl', 'wb') as f:
    pickle.dump(all_features, f)

print("Datasets limpios y features guardados en ../data/")
print("\\nResumen final:")
for name, ds in cleaned_datasets.items():
    print(f"\\n  {name}:")
    print(f"    Table A: {len(ds['tableA'])} productos")
    print(f"    Table B: {len(ds['tableB'])} productos")
    total_pairs = len(ds['train']) + len(ds['valid']) + len(ds['test'])
    match_count = sum(ds[s]['label'].sum() for s in ['train', 'valid', 'test'])
    print(f"    Pares totales: {total_pairs} ({match_count} matches, {total_pairs - match_count} no matches)")''')

# Build notebook
nb.cells = cells
nb.metadata = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    },
    'language_info': {
        'name': 'python',
        'version': '3.14.0'
    }
}

with open('notebooks/01_dataset_description.ipynb', 'w') as f:
    nbformat.write(nb, f)

print(f"Notebook generado con {len(cells)} celdas!")
print(f"  Markdown: {sum(1 for c in cells if c.cell_type == 'markdown')}")
print(f"  Code: {sum(1 for c in cells if c.cell_type == 'code')}")
