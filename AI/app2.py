from flask import Flask, request, jsonify, render_template, send_from_directory
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

app = Flask(__name__, 
    template_folder='templates',
    static_folder='static'
)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://your_username:your_password@localhost/pest_detection'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Database Models
# Update the database column types for MySQL compatibility
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

class DetectionAction(db.Model):
    __tablename__ = 'detection_actions'
    action_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    detection_id = db.Column(db.Integer, db.ForeignKey('pest_detections.detection_id'))
    action_taken = db.Column(db.String(255))
    action_date = db.Column(db.DateTime, default=datetime.utcnow)
    action_status = db.Column(db.String(50))
    notes = db.Column(db.Text)

class PredictionFeedback(db.Model):
    __tablename__ = 'prediction_feedback'
    feedback_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    detection_id = db.Column(db.Integer, db.ForeignKey('pest_detections.detection_id'))
    correct_prediction = db.Column(db.Boolean)
    actual_pest_type = db.Column(db.String(100))
    feedback_notes = db.Column(db.Text)
    feedback_date = db.Column(db.DateTime, default=datetime.utcnow)

# Your existing PestPredictor class remains the same
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
    return render_template('index.html')

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
        
        # Save the file
        filename = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Create detection record
        detection = PestDetection(
            image_path=filepath,
            ip_address=request.remote_addr,
            device_info=request.user_agent.string
        )
        db.session.add(detection)
        db.session.flush()
        
        # Process the image
        image = Image.open(filepath).convert('RGB')
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
        pest_info = predictor.get_pest_info(top_pest) if top_confidence > 50 else None
        
        # Convert image to base64 for display
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        db.session.commit()
        
        return jsonify({
            'detection_id': detection.detection_id,
            'predictions': [{'class': p[0], 'confidence': p[1]} for p in predictions],
            'pest_info': pest_info,
            'image': img_str
        })
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.json
        feedback = PredictionFeedback(
            detection_id=data['detection_id'],
            correct_prediction=data['correct_prediction'],
            actual_pest_type=data.get('actual_pest_type'),
            feedback_notes=data.get('feedback_notes')
        )
        db.session.add(feedback)
        db.session.commit()
        return jsonify({'message': 'Feedback submitted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    try:
        detections = PestDetection.query.order_by(PestDetection.submission_date.desc()).limit(10)
        history = []
        for detection in detections:
            predictions = PredictionResult.query.filter_by(detection_id=detection.detection_id).all()
            history.append({
                'detection_id': detection.detection_id,
                'submission_date': detection.submission_date.isoformat(),
                'image_path': detection.image_path,
                'predictions': [{
                    'pest_type': pred.pest_type,
                    'confidence_score': float(pred.confidence_score)
                } for pred in predictions]
            })
        return jsonify(history)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)