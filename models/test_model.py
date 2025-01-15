import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet18
import torch.nn as nn
import os
import json

class PestPredictor:
    def __init__(self, model_path, pest_info_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pest information
        with open(pest_info_path, 'r') as f:
            self.pest_info = json.load(f)
        
        # Define class names to match the trained model
        self.class_names = ['bedbug', 'cockroach', 'ants']  # Match your trained model classes
        
        # Initialize model
        self.model = resnet18()
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, len(self.class_names))
        
        # Load trained weights with weights_only=True for safety
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def get_pest_info(self, pest_type):
        """Get pest management information"""
        if pest_type in self.pest_info:
            info = self.pest_info[pest_type]
            return {
                'Scientific Name': info['scientific_name'],
                'Recommendations': info['recommendations'],
                'Control Measures': info['control_measures'],
                'Recommended Pesticides': info['pesticides']
            }
        return None
    
    def predict(self, image_path):
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get probabilities for all classes
            probs = probabilities[0].cpu().numpy()
            
            # Get predictions for all classes
            predictions = []
            for class_name, prob in zip(self.class_names, probs):
                predictions.append((class_name, prob * 100))
            
            # Sort by probability
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            return predictions

def main():
    try:
        # Get current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Paths to model and pest info
        model_path = os.path.join(current_dir, 'pest_classifier.pth')
        pest_info_path = os.path.join(current_dir, 'pest_info.json')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        if not os.path.exists(pest_info_path):
            raise FileNotFoundError(f"Pest info file not found at: {pest_info_path}")
        
        # Initialize predictor
        print("Loading model and pest information...")
        predictor = PestPredictor(model_path, pest_info_path)
        print(f"Model loaded successfully. Available classes: {predictor.class_names}")
        
        # Get image path from user
        while True:
            image_path = input("\nEnter the path to your test image (or 'q' to quit): ")
            
            if image_path.lower() == 'q':
                break
                
            if not os.path.exists(image_path):
                print(f"Error: Image not found at {image_path}")
                continue
                
            try:
                # Make prediction
                predictions = predictor.predict(image_path)
                
                print("\nPredictions:")
                print("-" * 50)
                for class_name, confidence in predictions:
                    print(f"{class_name}: {confidence:.2f}%")
                print("-" * 50)
                
                # Get pest information for top prediction if confidence is high enough
                top_pest, top_confidence = predictions[0]
                if top_confidence > 50 and top_pest != 'other':
                    pest_info = predictor.get_pest_info(top_pest)
                    if pest_info:
                        print(f"\nPest Information for {top_pest}:")
                        print("-" * 50)
                        print(f"Scientific Name: {pest_info['Scientific Name']}")
                        print("\nRecommendations:")
                        for rec in pest_info['Recommendations']:
                            print(f"- {rec}")
                        print("\nControl Measures:")
                        for measure in pest_info['Control Measures']:
                            print(f"- {measure}")
                        print("\nRecommended Pesticides:")
                        for pesticide in pest_info['Recommended Pesticides']:
                            print(f"- {pesticide}")
                        print("-" * 50)
                
            except Exception as e:
                print(f"Error processing image: {str(e)}")
                print("Please make sure the image is a valid image file.")
    
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure 'pest_classifier.pth' exists in the same directory as this script")
        print("2. Make sure 'pest_info.json' exists in the same directory as this script")
        print("3. Verify that your model was trained with these exact classes: ['bedbug', 'cockroach', 'ants']")
        print("4. Check that all required libraries are installed")

if __name__ == "__main__":
    main()