import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
import matplotlib.pyplot as plt
from pathlib import Path
import os

class PestClassifier:
    def __init__(self, num_classes, learning_rate=0.001):
        # Initialize model using ResNet18 with transfer learning
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet18(pretrained=True)
        
        # Modify the final layer for our number of classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        self.model = self.model.to(self.device)
        
        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def train_model(self, train_loader, val_loader, num_epochs):
        train_losses = []
        val_losses = []
        best_accuracy = 0.0
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            
            epoch_train_loss = running_loss / len(train_loader)
            train_losses.append(epoch_train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            epoch_val_loss = val_loss / len(val_loader)
            val_losses.append(epoch_val_loss)
            accuracy = 100. * correct / total
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(self.model.state_dict(), 'pest_classifier.pth')
            
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Training Loss: {epoch_train_loss:.4f}')
            print(f'Validation Loss: {epoch_val_loss:.4f}')
            print(f'Validation Accuracy: {accuracy:.2f}%\n')
        
        return train_losses, val_losses

def prepare_data(data_dir, batch_size=32):
    # Get the absolute path to the dataset directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, data_dir)
    
    print(f"Looking for dataset in: {dataset_path}")
    
    # Define data transforms with augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')
    
    # Updated required classes to match actual folder names
    required_classes = ['bedbugs', 'cockroach', 'ants']
    
    # Verify paths and structure
    for path in [train_path, val_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Directory not found at: {path}")
        for class_name in required_classes:
            class_path = os.path.join(path, class_name)
            if not os.path.exists(class_path):
                raise FileNotFoundError(f"Class directory '{class_name}' not found at: {class_path}")
    
    train_dataset = ImageFolder(train_path, transform=train_transform)
    val_dataset = ImageFolder(val_path, transform=val_transform)
    
    print(f"Found {len(train_dataset.classes)} classes: {train_dataset.classes}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, len(train_dataset.classes)

def main():
    data_dir = 'pest_dataset'
    
    try:
        print("Preparing to train pest classifier with the following classes:")
        print("1. bedbugs")  # Updated to match folder name
        print("2. cockroach")
        print("3. ants")    # Changed from "other" to match folder name
        print("\nEnsuring correct directory structure...")
        
        train_loader, val_loader, num_classes = prepare_data(data_dir)
        
        print("\nInitializing model...")
        model = PestClassifier(num_classes)
        print("Starting training...")
        train_losses, val_losses = model.train_model(train_loader, val_loader, num_epochs=20)
        
        # Save the final model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'pest_classifier.pth')
        torch.save(model.model.state_dict(), model_path)
        print(f"\nModel saved to: {model_path}")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nPlease ensure your directory structure looks like this:")
        print("pest_dataset/")
        print("├── train/")
        print("│   ├── bedbug/")    # Updated folder names
        print("│   │   └── images...")
        print("│   ├── cockroach/")
        print("│   │   └── images...")
        print("│   └── ants/")      # Updated folder name
        print("│       └── images...")
        print("└── val/")
        print("    ├── bedbug/")    # Updated folder names
        print("    │   └── images...")
        print("    ├── cockroach/")
        print("    │   └── images...")
        print("    └── ants/")      # Updated folder name
        print("        └── images...")

if __name__ == "__main__":
    main()