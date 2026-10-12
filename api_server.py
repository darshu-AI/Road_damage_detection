
import base64

import cv2
import numpy as np
import config
from flask import Flask, request, jsonify
from flask_cors import CORS

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
        return jsonify({
            "error": "Upload an image using the 'image' field"
        }), 400

    try:
        image_bytes = uploaded.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        image_bgr = cv2.imdecode(
            image_array, cv2.IMREAD_COLOR
        )

        if image_bgr is None:
            return jsonify({
                "error": "Invalid image file"
            }), 400

        if not config.MODEL_PATH.exists():
            return jsonify({
                "error": (
                    f"Model weights not found at "
                    f"{config.MODEL_PATH}"
                )
            }), 500

        result = predict_cracks(image_bgr)

        ok, encoded = cv2.imencode(
            ".jpg", result["annotated_bgr"]
        )

        if not ok:
            raise RuntimeError(
                "Could not encode annotated image"
            )

        response = {
            "detections": result["detections"],
            "class_counts": result["class_counts"],
            "total_cracks": result["total_cracks"],
            "average_confidence": result[
                "average_confidence"
            ],
            "has_detections": result["has_detections"],
            "annotated_image": base64.b64encode(
                encoded.tobytes()
            ).decode("utf-8")
        }

        return jsonify(response)

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": (
                "Prediction failed. "
                "Check the Render service logs."
            )
        }), 500
