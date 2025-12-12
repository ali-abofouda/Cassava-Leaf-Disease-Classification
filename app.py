"""
Cassava Leaf Disease Classification - Flask Web Application
============================================================
A professional web application for classifying cassava leaf diseases 
using a trained MobileNetV2 model with PyTorch.

Features:
- Image upload with drag & drop
- Real-time disease prediction
- Training dashboard with metrics
- Responsive Bootstrap 5 design
"""

import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import io
import base64

# =============================================================================
# Configuration
# =============================================================================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Paths
MODEL_PATH = os.path.join('results', 'best_mobilenet.pth')
RESULTS_JSON_PATH = os.path.join('results', 'training_results.json')

# Disease classes
CLASSES = ['CBB', 'CBSD', 'CGM', 'CMD', 'Healthy']
CLASS_NAMES = {
    'CBB': 'Cassava Bacterial Blight',
    'CBSD': 'Cassava Brown Streak Disease',
    'CGM': 'Cassava Green Mottle',
    'CMD': 'Cassava Mosaic Disease',
    'Healthy': 'Healthy Leaf'
}

CLASS_FULL_NAMES = {
    0: "Cassava Bacterial Blight (CBB)",
    1: "Cassava Brown Streak Disease (CBSD)",
    2: "Cassava Green Mottle (CGM)",
    3: "Cassava Mosaic Disease (CMD)",
    4: "Healthy"
}

CLASS_DESCRIPTIONS = {
    'CBB': 'A bacterial disease causing angular leaf spots, wilting, and gum exudates on stems. Can cause significant yield losses.',
    'CBSD': 'A viral disease causing yellow/brown streaks on leaves and stems. The root becomes dry and unusable.',
    'CGM': 'A viral disease causing mosaic patterns and leaf distortion. Often confused with CMD.',
    'CMD': 'The most devastating viral disease causing leaf curling, mosaic patterns, and stunted growth.',
    'Healthy': 'The leaf appears healthy with no visible disease symptoms. Good green coloration and normal leaf structure.'
}

CLASS_COLORS = {
    'CBB': '#dc3545',      # Red
    'CBSD': '#fd7e14',     # Orange
    'CGM': '#17a2b8',      # Cyan
    'CMD': '#6f42c1',      # Purple
    'Healthy': '#28a745'   # Green
}

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image transforms (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =============================================================================
# Model Loading
# =============================================================================
def load_model():
    """
    Load the trained MobileNetV2 model.
    
    The classifier head must match the architecture used during training:
    - Dropout(0.3)
    - Linear(1280, num_classes)
    """
    model = models.mobilenet_v2(weights=None)
    
    # Modify classifier to match training architecture
    # Simple classifier: Dropout + Linear (matching your training setup)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, len(CLASSES))
    )
    
    # Load trained weights
    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state_dict)
            print(f"✅ Model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"⚠️ Error loading model weights: {e}")
            print("   Attempting to load with flexible architecture...")
            # Try loading with flexible architecture
            try:
                model.classifier = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(model.last_channel, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.3),
                    nn.Linear(256, len(CLASSES))
                )
                state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
                model.load_state_dict(state_dict)
                print(f"✅ Model loaded with extended classifier from {MODEL_PATH}")
            except Exception as e2:
                print(f"❌ Failed to load model: {e2}")
    else:
        print(f"⚠️ Model file not found at {MODEL_PATH}")
        print("   Please ensure 'results/best_mobilenet.pth' exists.")
    
    model.to(DEVICE)
    model.eval()
    return model


def load_training_results():
    """Load training results from JSON file."""
    if os.path.exists(RESULTS_JSON_PATH):
        try:
            with open(RESULTS_JSON_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading training results: {e}")
    return None


# Load model at startup
print("="*60)
print("🌿 Loading Cassava Leaf Disease Classification Model...")
print("="*60)
model = load_model()

# =============================================================================
# Helper Functions
# =============================================================================
def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(image):
    """Make prediction on an image."""
    # Transform image
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item() * 100
    
    # Get all class probabilities
    all_probs = {CLASSES[i]: round(probabilities[i].item() * 100, 2) for i in range(len(CLASSES))}
    
    predicted_class = CLASSES[predicted_idx]
    
    return {
        'class': predicted_class,
        'class_name': CLASS_NAMES[predicted_class],
        'full_name': CLASS_FULL_NAMES[predicted_idx],
        'description': CLASS_DESCRIPTIONS[predicted_class],
        'confidence': round(confidence, 2),
        'color': CLASS_COLORS[predicted_class],
        'is_healthy': predicted_class == 'Healthy',
        'all_probabilities': all_probs
    }


# =============================================================================
# Routes
# =============================================================================
@app.route('/')
def index():
    """Home page - Image upload and prediction."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle single or multiple image upload and prediction."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    files = request.files.getlist('file')
    
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No file selected'}), 400
    
    results = []
    
    for file in files:
        if file.filename == '':
            continue
            
        if not allowed_file(file.filename):
            results.append({
                'filename': file.filename,
                'error': 'Invalid file type. Please upload PNG, JPG, or JPEG.',
                'success': False
            })
            continue
        
        try:
            # Read and process image
            image = Image.open(file.stream).convert('RGB')
            
            # Make prediction
            result = predict_image(image)
            
            # Convert image to base64 for display
            buffered = io.BytesIO()
            # Resize for thumbnail
            thumb = image.copy()
            thumb.thumbnail((400, 400))
            thumb.save(buffered, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            results.append({
                'filename': secure_filename(file.filename),
                'success': True,
                'prediction': result,
                'image': f'data:image/jpeg;base64,{img_base64}'
            })
        
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e),
                'success': False
            })
    
    # Return single result for backward compatibility if only one file
    if len(results) == 1:
        if results[0]['success']:
            return jsonify({
                'success': True,
                'prediction': results[0]['prediction'],
                'image': results[0]['image'],
                'filename': results[0]['filename']
            })
        else:
            return jsonify({'error': results[0]['error']}), 400
    
    # Return batch results
    return jsonify({
        'success': True,
        'batch': True,
        'results': results,
        'total': len(results),
        'successful': sum(1 for r in results if r['success']),
        'failed': sum(1 for r in results if not r['success'])
    })


@app.route('/dashboard')
def dashboard():
    """Dashboard page - Display training results and metrics."""
    results = load_training_results()
    return render_template('dashboard.html', results=results)


@app.route('/results/<path:filename>')
def serve_results(filename):
    """Serve static files from results folder (images, etc.)."""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)


@app.route('/api/results')
def api_results():
    """API endpoint to get training results as JSON."""
    results = load_training_results()
    if results:
        return jsonify(results)
    return jsonify({'error': 'No results found'}), 404


@app.route('/about')
def about():
    """About page."""
    return render_template('about.html')

# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == '__main__':
    # Create uploads folder if not exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    print("\n" + "="*60)
    print("🌿 Cassava Leaf Disease Classification Web App")
    print("="*60)
    print(f"📍 Device: {DEVICE}")
    print(f"🧠 Model: MobileNetV2")
    print(f"📁 Model Path: {MODEL_PATH}")
    print(f"🏷️  Classes: {CLASSES}")
    print("="*60)
    print("\n🚀 Starting server...")
    print("   Local:   http://localhost:5000")
    print("   Network: http://0.0.0.0:5000")
    print("\n📄 Routes:")
    print("   /           - Home (Image Upload)")
    print("   /predict    - Prediction API")
    print("   /dashboard  - Training Dashboard")
    print("   /about      - About Page")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
