
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from flask import Flask, jsonify, request

# Make the repository root importable.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import config
from core.detector import predict_cracks

app = Flask(__name__)

# Keep uploads below Vercel's request-body limit.
MAX_IMAGE_BYTES = 3 * 1024 * 1024
MAX_IMAGE_SIDE = 1280


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "Road Crack Detection API"
    })


@app.route("/api/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "Upload an image using the 'image' field."}), 400

    uploaded = request.files["image"]
    raw = uploaded.read(MAX_IMAGE_BYTES + 1)

    if not raw:
        return jsonify({"error": "The uploaded image is empty."}), 400

    if len(raw) > MAX_IMAGE_BYTES:
        return jsonify({
            "error": "Image is too large. Upload an image under 3 MB."
        }), 413

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "Invalid image file. Use JPG, PNG, or WEBP."}), 400

    # Resize large images to reduce inference time and memory usage.
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))

    image_rgb = np.array(image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    try:
        conf = float(request.form.get("conf", config.DEFAULT_CONF))
        iou = float(request.form.get("iou", config.DEFAULT_IOU))

        if not (0 <= conf <= 1 and 0 <= iou <= 1):
            return jsonify({
                "error": "conf and iou must be between 0 and 1."
            }), 400

        result = predict_cracks(image_bgr, conf=conf, iou=iou)

        # Return JSON-safe detection data and a displayable annotated image.
        annotated = result.get("annotated_rgb")
        if annotated is None:
            annotated = result.get("annotated_bgr")

            if annotated is not None:
                annotated = cv2.cvtColor(
                    annotated, cv2.COLOR_BGR2RGB
                )

        annotated_image = None
        if annotated is not None:
            annotated = np.asarray(annotated).astype(np.uint8)
            pil_image = Image.fromarray(annotated)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=85)
            annotated_image = (
                "data:image/jpeg;base64,"
                + base64.b64encode(buffer.getvalue()).decode("utf-8")
            )

        detections = result.get("detections", [])

        return jsonify({
            "success": True,
            "total_cracks": int(result.get("total_cracks", len(detections))),
            "class_counts": result.get("class_counts", {}),
            "average_confidence": float(
                result.get("average_confidence", 0.0)
            ),
            "has_detections": bool(
                result.get("has_detections", len(detections) > 0)
            ),
            "image_size": result.get("image_size"),
            "thresholds": {"confidence": conf, "iou": iou},
            "detections": detections,
            "annotated_image": annotated_image
        })

    except Exception as exc:
        app.logger.exception("Road crack inference failed")
        return jsonify({
            "error": "Inference failed. Check the Vercel function logs.",
            "detail": str(exc)[:500]
        }), 500


# Vercel discovers the Flask application as `app`.

