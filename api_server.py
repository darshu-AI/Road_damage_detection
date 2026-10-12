
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

from core.detector import predict_cracks

app = Flask(__name__)
CORS(app)

@app.get("/")
def health():
    return jsonify({
        "status": "online",
        "service": "Road Damage Detection API"
    })

@app.post("/predict")
def predict():
    uploaded = request.files.get("image")

    if uploaded is None:
        return jsonify({"error": "Upload an image using the 'image' field"}), 400

    try:
        image = Image.open(uploaded.stream).convert("RGB")
        result = predict_cracks(image)
        return jsonify({"result": result})
    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500
