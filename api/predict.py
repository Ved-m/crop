from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import os

app = Flask(__name__)

_model = None
_le = None
_model_name = None
_accuracy = None
_crops = None


def load_model():
    global _model, _le, _model_name, _accuracy, _crops
    if _model is None:
        paths = [
            'crop_model.pkl',
            '../crop_model.pkl',
            '/var/task/crop_model.pkl',
            os.path.join(os.path.dirname(__file__), '..', 'crop_model.pkl'),
        ]
        model_path = next((p for p in paths if os.path.exists(p)), None)
        if model_path is None:
            raise FileNotFoundError("crop_model.pkl not found")
        
        package = joblib.load(model_path)
        _model = package['model']
        _le = package['label_encoder']
        _model_name = package['model_name']
        _accuracy = package['accuracy']
        _crops = package['crops']
    return _model, _le, _model_name, _accuracy, _crops


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        model, le, model_name, accuracy, crops = load_model()
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        nitrogen = float(data['nitrogen'])
        phosphorus = float(data['phosphorus'])
        potassium = float(data['potassium'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])
        
        if not (0 <= ph <= 14):
            return jsonify({'success': False, 'error': 'Invalid pH value'}), 400
        
        features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
        prediction = model.predict(features)
        predicted_crop = le.inverse_transform(prediction)[0]
        
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


@app.route('/api/health', methods=['GET'])
def health():
    try:
        load_model()
        return jsonify({'status': 'healthy', 'model_loaded': True})
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500


@app.route('/', methods=['GET'])
def index():
    public_path = os.path.join(os.path.dirname(__file__), '..', 'public')
    return send_from_directory(public_path, 'index.html')


@app.route('/<path:path>', methods=['GET'])
def serve_file(path):
    if path == 'favicon.ico':
        return '', 204
    
    if path.startswith('api/'):
        return jsonify({'error': 'Not found'}), 404
    
    public_path = os.path.join(os.path.dirname(__file__), '..', 'public')
    
    try:
        return send_from_directory(public_path, path)
    except:
        return send_from_directory(public_path, 'index.html')
