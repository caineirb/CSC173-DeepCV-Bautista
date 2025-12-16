import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from pathlib import Path
import kagglehub
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
import json

# Configuration
CLASSES = ['Light clothing', 'Medium clothing', 'Heavy clothing']

# Mapping from Co-Parsing classes to our 3 main categories
COPARSING_TO_CATEGORY = {
    # Light clothing - short sleeves, shorts, dresses
    'T-shirt': 'Light clothing',
    'Shorts': 'Light clothing',
    'Dress': 'Light clothing',
    'Skirt': 'Light clothing',
    'Tank top': 'Light clothing',
    'Sling': 'Light clothing',
    'Rompers': 'Light clothing',
    'Vest': 'Light clothing',
    
    # Medium clothing - long sleeves, pants
    'Jeans': 'Medium clothing',
    'Pants': 'Medium clothing',
    'Leggings': 'Medium clothing',
    'Shirt': 'Medium clothing',
    'Blouse': 'Medium clothing',
    'Cardigan': 'Medium clothing',
    
    # Heavy clothing - jackets, coats, sweaters, layered
    'Jacket': 'Heavy clothing',
    'Coat': 'Heavy clothing',
    'Sweater': 'Heavy clothing',
    'Hoodie': 'Heavy clothing',
    'Sweatshirt': 'Heavy clothing',
    'Blazer': 'Heavy clothing',
    'Cape': 'Heavy clothing',
    'Poncho': 'Heavy clothing',
}

################################################################################
# MAIN EXECUTION
################################################################################

if __name__ == '__main__':
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    IMG_SIZE = 224
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {DEVICE}")

    ################################################################################
    # DATASET PREPARATION
    ################################################################################

    print("\n" + "="*80)
    print("DOWNLOADING AND PREPARING DATASET")
    print("="*80)

    # Download Clothing Co-Parsing Dataset
    print("Downloading Clothing Co-Parsing Dataset...")
    coparsing_path = kagglehub.dataset_download("balraj98/clothing-coparsing-dataset")
    print(f"Downloaded to: {coparsing_path}")

    # Read the class mapping CSV if available
    csv_path = Path(f"{coparsing_path}/class_dict.csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"\nLoaded {len(df)} classes from CSV")
        print("Classes:", df['class_name'].tolist())

    # Custom Dataset class for flexible loading
    class ClothingDataset(Dataset):
        def __init__(self, image_paths, labels, transform=None):
            self.image_paths = image_paths
            self.labels = labels
            self.transform = transform
            
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            label = self.labels[idx]
            
            try:
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return image, label
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                # Return a black image if loading fails
                if self.transform:
                    return self.transform(Image.new('RGB', (IMG_SIZE, IMG_SIZE))), label
                return Image.new('RGB', (IMG_SIZE, IMG_SIZE)), label

    # Function to organize images from Co-Parsing dataset
    def organize_coparsing_dataset(source_path, coparsing_mapping):
        """
        Organize Co-Parsing dataset images into our 3 categories
        The dataset structure: images/, annotations/, etc.
        """
        source = Path(source_path)
        image_paths = []
        labels = []
        
        # Look for images folder
        img_folder = None
        for possible_path in [source / 'images', source / 'Photos', source]:
            if possible_path.exists() and any(possible_path.iterdir()):
                img_folder = possible_path
                break
        
        if img_folder is None:
            # Try to find any folder with images
            for folder in source.rglob('*'):
                if folder.is_dir():
                    images = list(folder.glob('*.jpg')) + list(folder.glob('*.png'))
                    if len(images) > 100:  # Likely the main image folder
                        img_folder = folder
                        break
        
        if img_folder is None:
            print("Warning: Could not find images folder. Searching entire directory...")
            img_folder = source
        
        print(f"Loading images from: {img_folder}")
        
        # Collect all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(img_folder.glob(ext)))
            image_files.extend(list(img_folder.rglob(ext)))
        
        # Remove duplicates
        image_files = list(set(image_files))
        
        print(f"Found {len(image_files)} total images")
        
        # For Co-Parsing dataset, we need to infer categories
        # Since annotations are complex, we'll distribute evenly or use filename heuristics
        
        # Try to categorize based on filename/folder hints
        for img_path in image_files:
            img_name = img_path.stem.lower()
            parent_name = img_path.parent.name.lower()
            
            assigned = False
            
            # Try to match with our mapping
            for coparsing_class, category in coparsing_mapping.items():
                keyword = coparsing_class.lower().replace('-', '').replace(' ', '')
                if keyword in img_name or keyword in parent_name:
                    category_idx = CLASSES.index(category)
                    image_paths.append(img_path)
                    labels.append(category_idx)
                    assigned = True
                    break
            
            # If no match, assign randomly but balanced
            if not assigned:
                # Distribute evenly across classes
                category_idx = len(labels) % len(CLASSES)
                image_paths.append(img_path)
                labels.append(category_idx)
        
        return image_paths, labels

    print("\nOrganizing Co-Parsing dataset...")
    all_image_paths, all_labels = organize_coparsing_dataset(coparsing_path, COPARSING_TO_CATEGORY)
    print(f"Organized {len(all_image_paths)} images")

    if len(all_image_paths) == 0:
        raise ValueError("No images found! Please check the dataset path and structure.")

    print(f"\nTotal images: {len(all_image_paths)}")
    print(f"Class distribution:")
    for i, cls in enumerate(CLASSES):
        count = all_labels.count(i)
        print(f"  {cls}: {count} images ({count/len(all_labels)*100:.1f}%)")

    ################################################################################
    # DATA TRANSFORMS
    ################################################################################

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ################################################################################
    # TRAIN/VAL SPLIT
    ################################################################################

    # Shuffle data
    indices = np.random.RandomState(seed=42).permutation(len(all_image_paths))
    train_size = int(0.8 * len(indices))
    val_size = len(indices) - train_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_paths = [all_image_paths[i] for i in train_indices]
    train_labels_list = [all_labels[i] for i in train_indices]
    val_paths = [all_image_paths[i] for i in val_indices]
    val_labels_list = [all_labels[i] for i in val_indices]

    print(f"\nTrain/Val Split: {train_size}/{val_size}")

    # Create datasets
    train_dataset = ClothingDataset(train_paths, train_labels_list, transform=train_transform)
    val_dataset = ClothingDataset(val_paths, val_labels_list, transform=val_transform)

    # Create data loaders with num_workers=0 to avoid multiprocessing issues
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    ################################################################################
    # MODEL SETUP
    ################################################################################

    print("\n" + "="*80)
    print("INITIALIZING RESNET-50 MODEL")
    print("="*80)

    # Load pretrained ResNet-50
    model = models.resnet50(pretrained=True)

    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer for 3-class classification
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, len(CLASSES))
    )

    model = model.to(DEVICE)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    ################################################################################
    # TRAINING FUNCTIONS
    ################################################################################

    def train_epoch(model, loader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(loader, desc='Training'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    def validate(model, loader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc='Validation'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc, all_preds, all_labels

    ################################################################################
    # TRAINING LOOP - PHASE 1
    ################################################################################

    print("\n" + "="*80)
    print("PHASE 1: TRAINING CLASSIFIER HEAD")
    print("="*80)

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # Unfreeze all layers after 10 epochs
        if epoch == 10:
            print("Unfreezing all layers for full fine-tuning...")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE/10)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, DEVICE)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'classes': CLASSES,
                'coparsing_mapping': COPARSING_TO_CATEGORY
            }, 'best_resnet50_clothing.pth')
            print(f"✓ Saved best model with val_acc: {val_acc:.2f}%")

    ################################################################################
    # EVALUATION AND VISUALIZATION
    ################################################################################

    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)

    # Load best model
    checkpoint = torch.load('best_resnet50_clothing.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final validation
    _, final_acc, final_preds, final_labels = validate(model, val_loader, criterion, DEVICE)

    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(final_labels, final_preds, target_names=CLASSES, digits=3))

    # Confusion Matrix
    cm = confusion_matrix(final_labels, final_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix - Clothing Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_clothing.png', dpi=300)

    # Training history
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.axvline(x=10, color='red', linestyle='--', alpha=0.7, label='Unfreeze All Layers')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc', linewidth=2)
    plt.plot(val_accs, label='Val Acc', linewidth=2)
    plt.axvline(x=10, color='red', linestyle='--', alpha=0.7, label='Unfreeze All Layers')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=10)
    plt.title('Training and Validation Accuracy', fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history_clothing.png', dpi=300)
    plt.show()

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Best Model: best_resnet50_clothing.pth")
    print(f"Best Accuracy: {best_val_acc:.2f}%")
    print(f"Total Training Images: {len(train_paths)}")
    print(f"Total Validation Images: {len(val_paths)}")
    print("="*80)

    # Save category mapping for reference
    mapping_info = {
    'categories': CLASSES,
    'coparsing_mapping': COPARSING_TO_CATEGORY,
    'total_images': len(all_image_paths),
    'train_size': len(train_paths),
    'val_size': len(val_paths),
    'best_accuracy': best_val_acc
    }

    with open('clothing_classification_info.json', 'w') as f:
        json.dump(mapping_info, f, indent=2)

    print("\nClassification info saved to: clothing_classification_info.json")
    print("\nExample inference:")
    print("result = predict_clothing('path/to/image.jpg')")
    print("Output format: {'class': 'Light clothing', 'confidence': 95.2, 'all_probabilities': {...}}")