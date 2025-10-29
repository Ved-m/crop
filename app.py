from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Global variables for model caching
_model = None
_le = None
_model_name = None
_accuracy = None
_crops = None

def load_model():
    """Load model once and cache it"""
    global _model, _le, _model_name, _accuracy, _crops
    
    if _model is None:
        try:
            # Vercel stores files in /var/task/
            model_path = os.path.join(os.path.dirname(__file__), '..', 'crop_model.pkl')
            package = joblib.load(model_path)
            
            _model = package['model']
            _le = package['label_encoder']
            _model_name = package['model_name']
            _accuracy = package['accuracy']
            _crops = package['crops']
            
            print(f"✅ Model loaded: {_model_name}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    return _model, _le, _model_name, _accuracy, _crops

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        model, le, model_name, accuracy, crops = load_model()
        
        # Get data from request
        data = request.get_json()
        
        # Extract features
        nitrogen = float(data['nitrogen'])
        phosphorus = float(data['phosphorus'])
        potassium = float(data['potassium'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])
        
        # Validate ranges
        if not (0 <= ph <= 14):
            return jsonify({'success': False, 'error': 'pH must be 0-14'}), 400
        
        if not (0 <= humidity <= 100):
            return jsonify({'success': False, 'error': 'Humidity must be 0-100'}), 400
        
        if rainfall < 0:
            return jsonify({'success': False, 'error': 'Rainfall cannot be negative'}), 400
        
        # Create feature array
        features = np.array([[nitrogen, phosphorus, potassium, temperature, 
                            humidity, ph, rainfall]])
        
        # Make prediction
        prediction = model.predict(features)
        predicted_crop = le.inverse_transform(prediction)[0]
        
        # Get probabilities if available
        confidence = None
        top_predictions = None
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            confidence = float(np.max(proba) * 100)
            
            # Top 3 predictions
            top_indices = np.argsort(proba)[-3:][::-1]
            top_predictions = [
                {
                    'crop': le.inverse_transform([idx])[0],
                    'probability': float(proba[idx] * 100)
                }
                for idx in top_indices
            ]
        
        return jsonify({
            'success': True,
            'crop': predicted_crop,
            'confidence': confidence,
            'top_predictions': top_predictions,
            'model_info': {
                'name': model_name,
                'accuracy': accuracy
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error: {str(e)}'
        }), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    try:
        model, le, model_name, accuracy, crops = load_model()
        return jsonify({
            'success': True,
            'model_name': model_name,
            'accuracy': accuracy,
            'crops': crops
        })
    except:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 404

@app.route('/api/health', methods=['GET'])
def health():
    try:
        load_model()
        return jsonify({
            'status': 'healthy',
            'model_loaded': True
        })
    except:
        return jsonify({
            'status': 'unhealthy',
            'model_loaded': False
        }), 500

# Vercel serverless handler
def handler(request):
    with app.request_context(request.environ):
        return app.full_dispatch_request()