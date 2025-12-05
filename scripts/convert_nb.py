import json
import os

file_path = r"C:\Users\admin\Documents\Counting Activity for CV-CSI\wivi_project\src\multimodal_activity_recognition.ipynb"

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Iterate through cells to find the code
    for cell in data['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                # Fix Dataset Path
                if '/kaggle/input/csi-cv-combined-1/dataset_final' in line:
                    line = line.replace('/kaggle/input/csi-cv-combined-1/dataset_final', '../data/dataset_final')
                    # Remove newline to append comment correctly if needed, but simple replace is safer
                    # line = line.rstrip() + " # CHANGED: Point to your local data\n" 
                
                # Fix Kaggle Output Path logic if needed, but the original code had an 'else' block for local, so it might be fine.
                # Let's just ensure the local path is used.
                
                new_source.append(line)
            cell['source'] = new_source

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1)
    print("Successfully converted notebook.")
except Exception as e:
    print(f"Error: {e}")
