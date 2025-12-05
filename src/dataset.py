import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from .utils import hampel_filter, normalize_csi

# Configuration Constants (can be moved to a config file)
CSI_LENGTH = 100

class MultimodalDataset(Dataset):
    def __init__(self, dataset_path, split='train', transform=None, task='classification', verbose=True):
        self.dataset_path = dataset_path
        self.transform = transform
        self.task = task
        self.verbose = verbose
        
        # 1. Discover Classes
        self.classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        if self.verbose and split == 'train':
            print(f"Found {len(self.classes)} classes: {self.classes}")
        
        # 2. Collect All Samples from CSVs
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(dataset_path, cls_name)
            csi_dir = os.path.join(cls_dir, 'csi')
            img_dir = os.path.join(cls_dir, 'images')
            
            # Find the CSV file in csi folder
            if not os.path.exists(csi_dir):
                continue
            
            csv_files = [f for f in os.listdir(csi_dir) if f.endswith('.csv')]
            if not csv_files:
                if self.verbose: print(f"No CSV found in {csi_dir}")
                continue
                
            # Assuming one main CSV per class or read all
            for csv_file in csv_files:
                csv_path = os.path.join(csi_dir, csv_file)
                try:
                    df = pd.read_csv(csv_path)
                    # Expected columns: image_filename, normalized_csi_data, ...
                    if 'image_filename' not in df.columns or 'normalized_csi_data' not in df.columns:
                        if self.verbose: print(f"Skipping {csv_file}: missing columns")
                        continue
                    
                    for _, row in df.iterrows():
                        img_name = row['image_filename']
                        csi_str = row['normalized_csi_data']
                        
                        img_full_path = os.path.join(img_dir, img_name)
                        if os.path.exists(img_full_path):
                            self.samples.append({
                                'img_path': img_full_path,
                                'csi_data': csi_str,
                                'label': self.class_to_idx[cls_name],
                                'class_name': cls_name
                            })
                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")

        # 3. Split Data (80-10-10)
        np.random.seed(42)
        indices = np.random.permutation(len(self.samples))
        n_total = len(indices)
        train_end = int(n_total * 0.8)
        val_end = int(n_total * 0.9)
        
        if split == 'train':
            self.indices = indices[:train_end]
        elif split == 'val':
            self.indices = indices[train_end:val_end]
        else: # test
            self.indices = indices[val_end:]
            
        if self.verbose:
            print(f"[{split.upper()}] Loaded {len(self.indices)} samples out of {len(self.samples)} total.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        sample_info = self.samples[real_idx]
        
        # Load CSI
        try:
            # Parse string representation of list
            csi_raw = ast.literal_eval(sample_info['csi_data'])
            csi_data = np.array(csi_raw, dtype=np.float32)
            
            # Apply Hampel & Normalize
            csi_data = hampel_filter(csi_data, K=3)
            csi_data = normalize_csi(csi_data)
            
            # Pad or truncate
            if len(csi_data) > CSI_LENGTH:
                csi_data = csi_data[:CSI_LENGTH]
            else:
                pad_width = CSI_LENGTH - len(csi_data)
                csi_data = np.pad(csi_data, (0, pad_width), 'constant')
                
            csi_tensor = torch.tensor(csi_data, dtype=torch.float32)
            
        except Exception as e:
            # print(f"Error parsing CSI: {e}")
            csi_tensor = torch.zeros(CSI_LENGTH, dtype=torch.float32)

        # Load Image
        try:
            image = Image.open(sample_info['img_path']).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading Image {sample_info['img_path']}: {e}")
            image = torch.zeros((3, 224, 224), dtype=torch.float32)

        # Label
        label = torch.tensor(sample_info['label'], dtype=torch.long)
        
        return csi_tensor, image, label
