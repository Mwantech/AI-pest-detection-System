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
    def __init__(self, num_classes, class_to_idx=None, learning_rate=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet18(pretrained=True)
        
        # Modify the final layer for our number of classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        self.model = self.model.to(self.device)
        
        # Store class mapping
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()} if class_to_idx else None
        
        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train_model(self, train_loader, val_loader, num_epochs):
        train_losses = []
        val_losses = []
        best_accuracy = 0.0
        
        print(f"Training with class mapping: {self.class_to_idx}")
        
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
            
            # Save best model with class mapping
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                save_dict = {
                    'state_dict': self.model.state_dict(),
                    'class_to_idx': self.class_to_idx
                }
                torch.save(save_dict, 'pest_classifier.pth')
            
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Training Loss: {epoch_train_loss:.4f}')
            print(f'Validation Loss: {epoch_val_loss:.4f}')
            print(f'Validation Accuracy: {accuracy:.2f}%\n')
        
        return train_losses, val_losses

def prepare_data(data_dir, batch_size=32):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, data_dir)
    
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
    
    # Load training dataset first to establish class mapping
    train_dataset = ImageFolder(train_path, transform=train_transform)
    class_to_idx = train_dataset.class_to_idx
    print(f"Class to index mapping: {class_to_idx}")
    
    # Use same class mapping for validation dataset
    val_dataset = ImageFolder(val_path, transform=val_transform, 
                            target_transform=lambda x: list(class_to_idx.values()).index(x))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, len(train_dataset.classes), class_to_idx

def main():
    data_dir = 'pest_dataset'
    
    try:
        print("\nLoading and preparing datasets...")
        train_loader, val_loader, num_classes, class_to_idx = prepare_data(data_dir)
        
        print("\nInitializing model...")
        model = PestClassifier(num_classes, class_to_idx=class_to_idx)
        
        print("Starting training...")
        train_losses, val_losses = model.train_model(train_loader, val_loader, num_epochs=20)
        
        # Save the final model with class mapping
        save_dict = {
            'state_dict': model.model.state_dict(),
            'class_to_idx': model.class_to_idx
        }
        torch.save(save_dict, 'pest_classifier_final.pth')
        print(f"\nFinal model saved with class mapping")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nPlease ensure your directory structure is correct")

if __name__ == "__main__":
    main()