# Week 4 Capstone: Flask Prediction API
# Loads the serialized model and provides a /predict endpoint.

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "breast_cancer_model.joblib"

app = Flask(__name__)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model file not found. Run train_model.py before starting the API."
    )

model = joblib.load(MODEL_PATH)


@app.get("/")
def home():
    """Return basic API information."""
    return jsonify({
        "message": "Breast Cancer Prediction API",
        "endpoint": "/predict",
        "method": "POST"
    })


@app.post("/predict")
def predict():
    """Return a prediction for one observation."""
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict) or "features" not in payload:
        return jsonify({
            "error": "Request must contain a 'features' object."
        }), 400

    features = payload["features"]

    if not isinstance(features, dict):
        return jsonify({
            "error": "'features' must be a JSON object."
        }), 400

    expected_features = list(model.feature_names_in_)
    missing = [feature for feature in expected_features if feature not in features]

    if missing:
        return jsonify({
            "error": "Missing required features.",
            "missing_features": missing
        }), 400

    try:
        row = {feature: float(features[feature]) for feature in expected_features}
    except (TypeError, ValueError):
        return jsonify({
            "error": "All feature values must be numeric."
        }), 400

    input_data = pd.DataFrame([row], columns=expected_features)
    prediction = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]

    class_names = {
        0: "malignant",
        1: "benign"
    }

    return jsonify({
        "prediction": prediction,
        "prediction_label": class_names[prediction],
        "probability_malignant": round(float(probabilities[0]), 6),
        "probability_benign": round(float(probabilities[1]), 6)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
