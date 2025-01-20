
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
from flask_cors import CORS  # Add CORS support for Node.js integration

app = Flask(__name__, 
    template_folder='templates',
    static_folder='static'
)
CORS(app)  # Enable CORS for all routes

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/pest_detection'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# [Previous database models remain unchanged]
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

# [PestPredictor class remains unchanged]
class PestPredictor:
    def __init__(self, model_path, pest_info_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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

# Initialize predictor
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

# Updated routes for Node.js integration
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Handle both form-data and JSON requests
        if request.is_json:
            # Handle base64 encoded image from JSON
            data = request.get_json()
            if 'image' not in data:
                return jsonify({'error': 'No image provided in JSON'}), 400
            
            # Decode base64 image
            try:
                image_data = base64.b64decode(data['image'].split(',')[1] if ',' in data['image'] else data['image'])
                image = Image.open(BytesIO(image_data)).convert('RGB')
            except Exception as e:
                return jsonify({'error': f'Invalid image data: {str(e)}'}), 400
            
            # Generate filename for base64 image
            filename = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jpg"
            
        else:
            # Handle multipart form data
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type'}), 400
            
            filename = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
            image = Image.open(file).convert('RGB')

        # Save the image
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(filepath)
        
        # Create detection record
        detection = PestDetection(
            image_path=filepath,
            ip_address=request.remote_addr,
            device_info=request.headers.get('User-Agent', '')
        )
        db.session.add(detection)
        db.session.flush()
        
        # Process the image
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
        
        # Convert image to base64 for response
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'detection_id': detection.detection_id,
            'predictions': [{'class': p[0], 'confidence': p[1]} for p in predictions],
            'pest_info': pest_info,
            'image': img_str
        })
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/feedback', methods=['POST'])
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
        return jsonify({'success': True, 'message': 'Feedback submitted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        detections = PestDetection.query.order_by(
            PestDetection.submission_date.desc()
        ).paginate(page=page, per_page=per_page)
        
        history = []
        for detection in detections.items:
            predictions = PredictionResult.query.filter_by(detection_id=detection.detection_id).all()
            
            # Convert image to base64
            try:
                with open(detection.image_path, 'rb') as img_file:
                    img_data = base64.b64encode(img_file.read()).decode()
            except Exception as e:
                img_data = None
            
            history.append({
                'detection_id': detection.detection_id,
                'submission_date': detection.submission_date.isoformat(),
                'image': img_data,
                'predictions': [{
                    'pest_type': pred.pest_type,
                    'confidence_score': float(pred.confidence_score)
                } for pred in predictions]
            })
        
        return jsonify({
            'success': True,
            'data': history,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total_pages': detections.pages,
                'total_items': detections.total
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)















