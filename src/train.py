import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms

from .dataset import MultimodalDataset
from .model import FusionModel

# Configuration
DATASET_PATH = "../data/data_activity"
OUTPUT_BASE_PATH = "../outputs"
MODEL_SAVE_PATH = os.path.join(OUTPUT_BASE_PATH, "models")
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(mode='fusion'):
    print(f"\n{'='*40}")
    print(f"STARTING TRAINING MODE: {mode.upper()}")
    print(f"{'='*40}")
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset & Dataloader
    # Note: In a real script, you might want to split this properly or use the split logic from the dataset class
    full_dataset = MultimodalDataset(DATASET_PATH, split='train', transform=transform, verbose=True)
    
    # Simple split for demonstration (using the dataset's internal split logic would be better if exposed)
    # Here we just use the dataset as is (which is 80% of total if split='train')
    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # Validation set
    val_dataset = MultimodalDataset(DATASET_PATH, split='val', transform=transform, verbose=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Model
    num_classes = len(full_dataset.classes)
    model = FusionModel(num_classes=num_classes, mode=mode).to(DEVICE)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training Loop
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for csi, img, labels in progress_bar:
            csi, img, labels = csi.to(DEVICE), img.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(csi, img)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({'loss': loss.item()})
            
        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for csi, img, labels in val_loader:
                csi, img, labels = csi.to(DEVICE), img.to(DEVICE), labels.to(DEVICE)
                outputs = model(csi, img)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, f'best_model_{mode}.pth'))
            print(f"Saved best model with acc: {best_val_acc:.2f}%")

    print(f"Training complete. Best Val Acc: {best_val_acc:.2f}%")

if __name__ == "__main__":
    # You can change the mode here or use argparse
    train_model(mode='fusion')
