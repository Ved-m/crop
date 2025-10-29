from flask import Flask
from .predict import predict_bp

def create_app():
    app = Flask(__name__)
    app.register_blueprint(predict_bp, url_prefix='/api')
    return app
