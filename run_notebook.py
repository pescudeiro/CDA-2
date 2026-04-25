#!/usr/bin/env python3
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

with open('notebooks/01_dataset_description.ipynb') as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=180, kernel_name='python3')
try:
    ep.preprocess(nb, {'metadata': {'path': 'notebooks'}})
    print('Notebook executed successfully!')
    # Save executed notebook
    with open('notebooks/01_dataset_description_executed.ipynb', 'w') as f:
        nbformat.write(nb, f)
except Exception as e:
    print(f'Error: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()
