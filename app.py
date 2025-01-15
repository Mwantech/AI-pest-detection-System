from flask import Flask, request, jsonify, render_template
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import os
import json
import base64
from io import BytesIO

app = Flask(__name__, 
    template_folder='templates',  # Explicitly set template folder
    static_folder='static'        # Explicitly set static folder
)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class PestPredictor:
    def __init__(self, model_path, pest_info_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pest information
        with open(pest_info_path, 'r') as f:
            self.pest_info = json.load(f)
        
        # Load the saved model data with weights_only=True
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        
        # Get class mapping from saved model
        self.class_to_idx = checkpoint['class_to_idx']
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.class_names = list(self.class_to_idx.keys())
        
        # Initialize model with updated weights parameter
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, len(self.class_names))
        
        # Load the trained state dict
        self.model.load_state_dict(checkpoint['state_dict'])
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
        if pest_type in self.pest_info:
            info = self.pest_info[pest_type]
            return {
                'scientific_name': info['scientific_name'],
                'recommendations': info['recommendations'],
                'control_measures': info['control_measures'],
                'pesticides': info['pesticides']
            }
        return None
    
    def predict(self, image):
        # Transform image
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get probabilities for all classes
            probs = probabilities[0].cpu().numpy()
            
            # Get predictions for all classes using correct class mapping
            predictions = []
            for idx, prob in enumerate(probs):
                class_name = self.idx_to_class[idx]
                predictions.append((class_name, float(prob * 100)))
            
            # Sort by probability
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            return predictions

# Initialize the model
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'pest_classifier.pth')
    pest_info_path = os.path.join(current_dir, 'pest_info.json')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(pest_info_path):
        raise FileNotFoundError(f"Pest info file not found at {pest_info_path}")
        
    predictor = PestPredictor(model_path, pest_info_path)
except Exception as e:
    print(f"Error initializing predictor: {str(e)}")
    raise

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Error rendering template: {str(e)}")
        return str(e), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Read and process the image
        image = Image.open(file.stream).convert('RGB')
        
        # Get predictions
        predictions = predictor.predict(image)
        
        # Get pest info for top prediction
        top_pest, top_confidence = predictions[0]
        pest_info = predictor.get_pest_info(top_pest) if top_confidence > 50 else None
        
        # Convert image to base64 for display
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Prepare response
        response = {
            'predictions': [{'class': p[0], 'confidence': p[1]} for p in predictions],
            'pest_info': pest_info if pest_info else None,
            'image': img_str
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)