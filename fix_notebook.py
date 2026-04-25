import nbformat

# Read the notebook
with open('notebooks/01_dataset_description.ipynb') as f:
    nb = nbformat.read(f, as_version=4)

# Fix the problematic cell (examples of pairs)
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and 'Ejemplos de Pares Match' in ''.join(nb.cells[i-1]['source'] if i > 0 else []):
        cell.source = '''for name, ds in datasets.items():
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
        print(f"\\n  Amazon/TableA: {row[t1_col][:120]}")
        print(f"  Google/TableB: {row[t2_col][:120]}")
    
    # Ejemplos de NO MATCH
    no_matches = train[train['label'] == 0].head(3)
    print("\\n--- Ejemplos de NO MATCH (productos diferentes) ---")
    for _, row in no_matches.iterrows():
        print(f"\\n  Amazon/TableA: {row[t1_col][:120]}")
        print(f"  Google/TableB: {row[t2_col][:120]}")
    
    # Ejemplos de falsos positivos potenciales
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
    
    print()'''
        break

# Fix the generate_features function
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and 'def generate_features' in ''.join(cell['source']):
        cell.source = '''def generate_features(ds, name):
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
            print(f"    {col}: {corr:.4f}")'''
        break

# Save the fixed notebook
with open('notebooks/01_dataset_description.ipynb', 'w') as f:
    nbformat.write(nb, f)

print("Notebook fixed successfully!")
