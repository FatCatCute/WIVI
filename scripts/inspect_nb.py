import json

file_path = r"C:\Users\admin\Documents\Counting Activity for CV-CSI\wivi_project\src\multimodal_activity_recognition.ipynb"

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "class" in source and "Dataset" in source:
            print(f"--- Cell {i} ---")
            print(source[:500]) # Print first 500 chars to identify
        if "DATASET_PATH =" in source:
            print(f"--- Cell {i} (Config) ---")
            print(source)
