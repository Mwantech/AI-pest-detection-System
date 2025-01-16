import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
from pathlib import Path
import os
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import pandas as pd

class PestClassifier:
    def __init__(self, num_classes, learning_rate=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize model using ResNet18 with proper weights
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Freeze early layers
        for param in list(self.model.parameters())[:-4]:
            param.requires_grad = False
        
        # Modify the final layer for our number of classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
        self.model = self.model.to(self.device)
        
        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam([
            {'params': self.model.fc.parameters(), 'lr': learning_rate},
            {'params': self.model.layer4.parameters(), 'lr': learning_rate/10}
        ])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', 
                                                            factor=0.5, patience=3, verbose=True)
    
    def train_model(self, train_loader, val_loader, num_epochs, class_names):
        train_losses = []
        val_losses = []
        best_accuracy = 0.0
        epochs_without_improvement = 0
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            epoch_train_loss = running_loss / len(train_loader)
            train_accuracy = 100. * correct / total
            train_losses.append(epoch_train_loss)
            
            # Validation phase
            val_loss, val_accuracy, confusion_mat = self.evaluate(val_loader)
            val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_accuracy)
            
            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(self.model.state_dict(), 'best_pest_classifier.pth')
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Print epoch statistics
            epoch_time = time.time() - start_time
            print(f'\nEpoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s')
            print(f'Training Loss: {epoch_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
            print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')
            
            # Print confusion matrix every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.plot_confusion_matrix(confusion_mat, class_names, epoch + 1)
            
            # Early stopping
            if epochs_without_improvement >= 10:
                print("Early stopping triggered!")
                break
        
        return train_losses, val_losses
    
    def evaluate(self, loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100. * correct / total
        avg_loss = running_loss / len(loader)
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        return avg_loss, accuracy, conf_matrix
    
    def plot_confusion_matrix(self, conf_matrix, class_names, epoch):
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'confusion_matrix_epoch_{epoch}.png')
        plt.close()

def prepare_data(data_dir, batch_size=32):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, data_dir)
    
    print(f"Looking for dataset in: {dataset_path}")
    
    # Enhanced data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Larger initial size for better cropping
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomCrop(224),
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
    
    # Print dataset statistics
    def print_dataset_stats(path, name):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} directory not found at: {path}")
        
        print(f"\n{name} Dataset Statistics:")
        print("-" * 30)
        for class_name in os.listdir(path):
            class_path = os.path.join(path, class_name)
            if os.path.isdir(class_path):
                num_images = len([f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
                print(f"{class_name}: {num_images} images")
    
    print_dataset_stats(train_path, "Training")
    print_dataset_stats(val_path, "Validation")
    
    train_dataset = ImageFolder(train_path, transform=train_transform)
    val_dataset = ImageFolder(val_path, transform=val_transform)
    
    class_names = train_dataset.classes
    print(f"\nClasses: {class_names}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    
    return train_loader, val_loader, len(class_names), class_names

def main():
    data_dir = 'pest_dataset'
    
    try:
        print("Preparing to train pest classifier...")
        train_loader, val_loader, num_classes, class_names = prepare_data(data_dir)
        
        print("\nInitializing model...")
        model = PestClassifier(num_classes)
        
        print("\nStarting training...")
        train_losses, val_losses = model.train_model(train_loader, val_loader, num_epochs=50, class_names=class_names)
        
        # Save the final model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'pest_classifier.pth')
        torch.save(model.model.state_dict(), model_path)
        print(f"\nModel saved to: {model_path}")
        
        # Plot training history
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('training_history.png')
        plt.close()
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nPlease ensure your directory structure looks like this:")
        print("pest_dataset/")
        print("├── train/")
        print("│   ├── bedbug/")
        print("│   ├── cockroach/")
        print("│   └── ants/")
        print("└── val/")
        print("    ├── bedbug/")
        print("    ├── cockroach/")
        print("    └── ants/")

if __name__ == "__main__":
    main()