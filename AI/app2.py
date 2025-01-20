from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import os
import json
import base64
from io import BytesIO
from flask_cors import CORS
import logging
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Enhanced CORS configuration
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://192.168.151.58:8081", "http://localhost:8081", "exp://localhost:8081"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Range", "X-Total-Count"],
        "supports_credentials": True,
        "max_age": 600
    }
})

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/pest_detection'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Upload configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

db = SQLAlchemy(app)

# Database Models
class PestDetection(db.Model):
    __tablename__ = 'pest_detections'
    detection_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    image_path = db.Column(db.String(255), nullable=False)
    submission_date = db.Column(db.DateTime, default=datetime.utcnow)
    ip_address = db.Column(db.String(45))
    device_info = db.Column(db.String(255))

class PredictionResult(db.Model):
    __tablename__ = 'prediction_results'
    prediction_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    detection_id = db.Column(db.Integer, db.ForeignKey('pest_detections.detection_id'))
    pest_type = db.Column(db.String(100), nullable=False)
    confidence_score = db.Column(db.Float(precision=5), nullable=False)
    prediction_timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class PestPredictor:
    def __init__(self, model_path, pest_info_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        with open(pest_info_path, 'r') as f:
            self.pest_info = json.load(f)
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        
        self.class_to_idx = checkpoint['class_to_idx']
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.class_names = list(self.class_to_idx.keys())
        
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, len(self.class_names))
        
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("Model loaded successfully")

    def predict(self, image):
        try:
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                probs = probabilities[0].cpu().numpy()
                
                predictions = []
                for idx, prob in enumerate(probs):
                    class_name = self.idx_to_class[idx]
                    predictions.append((class_name, float(prob * 100)))
                
                predictions.sort(key=lambda x: x[1], reverse=True)
                return predictions
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

    def get_pest_info(self, pest_type):
        return self.pest_info.get(pest_type)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize predictor
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'pest_classifier.pth')
    pest_info_path = os.path.join(current_dir, 'pest_info.json')
    predictor = PestPredictor(model_path, pest_info_path)
except Exception as e:
    logger.error(f"Failed to initialize predictor: {str(e)}")
    raise

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        logger.info("Received prediction request")
        
        # Check if the post request has the file part
        if 'image' not in request.files:
            logger.warning("No file part in request")
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['image']
        
        # If user does not select file
        if file.filename == '':
            logger.warning("No selected file")
            return jsonify({'error': 'No selected file'}), 400
            
        if not allowed_file(file.filename):
            logger.warning(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type'}), 400

        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Open and verify the image
        try:
            image = Image.open(file).convert('RGB')
            image.save(filepath)
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
            return jsonify({'error': 'Invalid image file'}), 400

        # Create detection record
        detection = PestDetection(
            image_path=filepath,
            ip_address=request.remote_addr,
            device_info=request.headers.get('User-Agent', '')
        )
        db.session.add(detection)
        db.session.flush()

        # Make prediction
        predictions = predictor.predict(image)

        # Store predictions
        for pest_type, confidence in predictions:
            pred_result = PredictionResult(
                detection_id=detection.detection_id,
                pest_type=pest_type,
                confidence_score=confidence
            )
            db.session.add(pred_result)

        # Get pest info for top prediction
        top_pest, top_confidence = predictions[0]
        pest_info = predictor.get_pest_info(top_pest)

        db.session.commit()

        response = {
            'success': True,
            'predictions': [{'class': p[0], 'confidence': p[1]} for p in predictions],
            'pest_info': pest_info
        }

        logger.info(f"Successful prediction for detection_id: {detection.detection_id}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat()
    })

# Error handler for large files
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large'}), 413

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)