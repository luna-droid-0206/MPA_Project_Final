"""
Professional Flask Web App for SimCLR Self-Supervised Learning
Provides a complete interface for model exploration, inference, and visualization
"""

import os
import json
import io
import base64
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import (
    CHECKPOINT_DIR, RESULTS_DIR, DATA_DIR,
    PRETRAINED_CHECKPOINT, SUPERVISED_CHECKPOINT, LINEAR_EVAL_CHECKPOINT,
    NUM_CLASSES, CIFAR10_MEAN, CIFAR10_STD, BATCH_SIZE
)
from models.encoder import Encoder

# ============================================================================
# Flask App Configuration
# ============================================================================

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# Device and Model Setup
# ============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]

# ============================================================================
# Model Loading
# ============================================================================

def _remap_state_dict_keys(state_dict):
    """Remap checkpoint keys to match model structure"""
    new_state_dict = {}
    
    for key, value in state_dict.items():
        # Handle encoder.encoder.* -> 0.encoder.*
        if key.startswith('encoder.encoder.'):
            new_key = '0.' + key[8:]  # Remove 'encoder.'
            new_state_dict[new_key] = value
        # Handle classifier.* -> 1.*
        elif key.startswith('classifier.'):
            new_key = '1.' + key[11:]  # Replace 'classifier.' with '1.'
            new_state_dict[new_key] = value
        # Handle direct encoder keys (just add 0. prefix)
        elif key.startswith('encoder.') and not key.startswith('encoder.encoder.'):
            new_key = '0.' + key
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    return new_state_dict


def load_encoder(checkpoint_path):
    """Load pretrained encoder"""
    if not os.path.exists(checkpoint_path):
        return None
    
    encoder = Encoder(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Handle checkpoint format - try different keys
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'encoder' in checkpoint:
            state_dict = checkpoint['encoder']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    encoder.load_state_dict(state_dict, strict=False)
    encoder = encoder.to(DEVICE)
    encoder.eval()
    return encoder


def load_supervised_model():
    """Load full supervised model"""
    if not os.path.exists(SUPERVISED_CHECKPOINT):
        return None
    
    model = Encoder(pretrained=False)
    fc = torch.nn.Linear(512, NUM_CLASSES)
    model = torch.nn.Sequential(model, fc)
    
    checkpoint = torch.load(SUPERVISED_CHECKPOINT, map_location=DEVICE)
    
    # Handle checkpoint format - try different keys
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Try to remap keys if they don't match
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        state_dict = _remap_state_dict_keys(state_dict)
        model.load_state_dict(state_dict, strict=False)
    
    model = model.to(DEVICE)
    model.eval()
    return model


def load_linear_eval_model():
    """Load linear evaluation model"""
    if not os.path.exists(LINEAR_EVAL_CHECKPOINT):
        return None
    
    encoder = Encoder(pretrained=False)
    fc = torch.nn.Linear(512, NUM_CLASSES)
    model = torch.nn.Sequential(encoder, fc)
    
    checkpoint = torch.load(LINEAR_EVAL_CHECKPOINT, map_location=DEVICE)
    
    # Handle checkpoint format - try different keys
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Try to remap keys if they don't match
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        state_dict = _remap_state_dict_keys(state_dict)
        model.load_state_dict(state_dict, strict=False)
    
    model = model.to(DEVICE)
    model.eval()
    return model


# ============================================================================
# Utility Functions
# ============================================================================

def get_model_info():
    """Get information about available models"""
    models_info = {
        'SimCLR Pretrained': {
            'path': PRETRAINED_CHECKPOINT,
            'exists': os.path.exists(PRETRAINED_CHECKPOINT),
            'description': 'Self-supervised pretrained encoder trained with contrastive learning (NT-Xent loss)',
            'type': 'encoder'
        },
        'Supervised Baseline': {
            'path': SUPERVISED_CHECKPOINT,
            'exists': os.path.exists(SUPERVISED_CHECKPOINT),
            'description': 'Standard supervised ResNet-18 trained on labeled CIFAR-10',
            'type': 'classifier'
        },
        'Linear Evaluation': {
            'path': LINEAR_EVAL_CHECKPOINT,
            'exists': os.path.exists(LINEAR_EVAL_CHECKPOINT),
            'description': 'Linear classifier on frozen SimCLR encoder features',
            'type': 'classifier'
        }
    }
    return models_info


def prepare_image(image_path_or_file):
    """Prepare image for inference"""
    if isinstance(image_path_or_file, str):
        img = Image.open(image_path_or_file).convert('RGB')
    else:
        img = Image.open(image_path_or_file).convert('RGB')
    
    # Resize to 32x32 (CIFAR-10 size)
    img = img.resize((32, 32), Image.Resampling.LANCZOS)
    img_array = np.array(img) / 255.0
    
    # Normalize
    mean = np.array(CIFAR10_MEAN)
    std = np.array(CIFAR10_STD)
    img_array = (img_array - mean) / std
    
    # Convert to tensor
    img_tensor = torch.FloatTensor(img_array).permute(2, 0, 1).unsqueeze(0)
    return img_tensor.to(DEVICE), img


def get_predictions(model, image_tensor):
    """Get predictions from model"""
    with torch.no_grad():
        output = model(image_tensor)
        if isinstance(output, tuple):
            output = output[0]
        probs = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_class].item()
    
    return predicted_class, confidence, probs.cpu().numpy()[0]


def get_features(encoder, image_tensor):
    """Extract features from encoder"""
    with torch.no_grad():
        features = encoder(image_tensor)
        features = features.view(features.size(0), -1)
    return features.cpu().numpy()[0]


# ============================================================================
# Flask Routes
# ============================================================================

@app.route('/')
def index():
    """Home page"""
    models_info = get_model_info()
    available_models = {k: v for k, v in models_info.items() if v['exists']}
    
    return render_template('index.html', 
                         models_info=available_models,
                         device=str(DEVICE))


@app.route('/api/models')
def api_models():
    """API endpoint for model information"""
    models_info = get_model_info()
    return jsonify(models_info)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page and API"""
    if request.method == 'GET':
        return render_template('predict.html', classes=CIFAR10_CLASSES)
    
    # Handle POST requests
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    model_choice = request.form.get('model', 'Supervised Baseline')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Prepare image
        img_tensor, img_pil = prepare_image(file)
        
        # Load appropriate model
        if model_choice == 'Supervised Baseline':
            model = load_supervised_model()
        elif model_choice == 'Linear Evaluation':
            model = load_linear_eval_model()
        else:
            return jsonify({'error': f'Unknown model: {model_choice}'}), 400
        
        if model is None:
            return jsonify({'error': f'Model not found: {model_choice}'}), 404
        
        # Get predictions
        predicted_class, confidence, probs = get_predictions(model, img_tensor)
        
        # Save uploaded image
        filename = secure_filename(f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img_pil.save(file_path)
        
        # Prepare response
        result = {
            'model': model_choice,
            'predicted_class': CIFAR10_CLASSES[predicted_class],
            'class_id': predicted_class,
            'confidence': float(confidence),
            'all_probabilities': {
                CIFAR10_CLASSES[i]: float(probs[i]) 
                for i in range(len(CIFAR10_CLASSES))
            },
            'image_path': f'/static/uploads/{filename}'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/features', methods=['GET', 'POST'])
def features():
    """Feature extraction page"""
    if request.method == 'GET':
        return render_template('features.html')
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    try:
        encoder = load_encoder(PRETRAINED_CHECKPOINT)
        if encoder is None:
            return jsonify({'error': 'SimCLR model not found'}), 404
        
        # Prepare image
        img_tensor, img_pil = prepare_image(file)
        
        # Extract features
        features = get_features(encoder, img_tensor)
        
        result = {
            'feature_dim': int(features.shape[0]),
            'features_mean': float(np.mean(features)),
            'features_std': float(np.std(features)),
            'features_min': float(np.min(features)),
            'features_max': float(np.max(features)),
            'features': features.tolist()
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/evaluation')
def evaluation():
    """Evaluation results page"""
    results_file = os.path.join(CHECKPOINT_DIR, 'evaluation_results.json')
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            eval_results = json.load(f)
    else:
        eval_results = None
    
    return render_template('evaluation.html', results=eval_results)


@app.route('/api/evaluation-results')
def api_evaluation_results():
    """API endpoint for evaluation results"""
    results_file = os.path.join(CHECKPOINT_DIR, 'evaluation_results.json')
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        return jsonify(results)
    
    return jsonify({'error': 'No evaluation results found'}), 404


@app.route('/documentation')
def documentation():
    """Documentation page"""
    # Read README
    readme_path = Path(__file__).parent / 'README.md'
    readme_content = ""
    if readme_path.exists():
        with open(readme_path, 'r') as f:
            readme_content = f.read()
    
    return render_template('documentation.html', readme=readme_content)


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': str(DEVICE),
        'models': get_model_info()
    })


# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return render_template('500.html', error=str(error)), 500


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print(f"Starting SimCLR Web App...")
    print(f"Device: {DEVICE}")
    print(f"Available Models: {get_model_info()}")
    
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        use_reloader=True
    )
