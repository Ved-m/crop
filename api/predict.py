from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
predict_bp = Blueprint('predict_bp', __name__)

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
            possible_paths = [
                'crop_model.pkl',
                '../crop_model.pkl',
                '/var/task/crop_model.pkl',
                os.path.join(os.path.dirname(__file__), '..', 'crop_model.pkl'),
            ]
            
            model_path = next((p for p in possible_paths if os.path.exists(p)), None)
            if model_path is None:
                raise FileNotFoundError("crop_model.pkl not found")

            print(f"Loading model from: {model_path}")
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
        data = request.get_json()

        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        # Extract and validate inputs
        nitrogen = float(data['nitrogen'])
        phosphorus = float(data['phosphorus'])
        potassium = float(data['potassium'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])

        if not (0 <= ph <= 14):
            return jsonify({'success': False, 'error': 'pH must be 0-14'}), 400
        if not (0 <= humidity <= 100):
            return jsonify({'success': False, 'error': 'Humidity must be 0-100'}), 400
        if rainfall < 0:
            return jsonify({'success': False, 'error': 'Rainfall cannot be negative'}), 400

        # Predict
        features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
        prediction = model.predict(features)
        predicted_crop = le.inverse_transform(prediction)[0]

        confidence, top_predictions = None, None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            confidence = float(np.max(proba) * 100)
            top_indices = np.argsort(proba)[-3:][::-1]
            top_predictions = [
                {'crop': le.inverse_transform([idx])[0], 'probability': float(proba[idx] * 100)}
                for idx in top_indices
            ]

        return jsonify({
            'success': True,
            'crop': predicted_crop,
            'confidence': confidence,
            'top_predictions': top_predictions,
            'model_info': {'name': model_name, 'accuracy': accuracy}
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': traceback.format_exc()}), 500


@app.route('/api/model-info', methods=['GET'])
def model_info():
    try:
        model, le, model_name, accuracy, crops = load_model()
        return jsonify({'success': True, 'model_name': model_name, 'accuracy': accuracy, 'crops': crops})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    try:
        load_model()
        return jsonify({'status': 'healthy', 'model_loaded': True})
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'model_loaded': False, 'error': str(e)}), 500


# ✅ Vercel entry point
# Just expose the Flask app object
app = app
