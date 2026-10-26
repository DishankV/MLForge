
from pathlib import Path
import math
import json

import joblib
from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parents[1]
MODEL_VERSION = "churn-logreg-v1"
VERSION_DIR = ROOT / "ml" / "registry" / "models" / MODEL_VERSION

MODEL_PATH = VERSION_DIR / "model.joblib"
METADATA_PATH = VERSION_DIR / "metadata.json"

THRESHOLD = 0.50

FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

NUMERIC_FEATURES = {
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
}

app = Flask(__name__)

# Only serve models that passed the promotion gate.
if not MODEL_PATH.is_file() or not METADATA_PATH.is_file():
    raise FileNotFoundError(
        "Registered model or metadata is missing. "
        "Register and approve a model first."
    )

with METADATA_PATH.open("r", encoding="utf-8") as file:
    metadata = json.load(file)

if metadata.get("status") != "approved":
    raise RuntimeError(
        f"Model {MODEL_VERSION} is not approved for deployment."
    )

model = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_version": MODEL_VERSION,
        "model_loaded": model is not None,
    }), 200


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({
            "error": "Request body must be a JSON object."
        }), 400

    missing = sorted(set(FEATURES) - set(payload))
    extra = sorted(set(payload) - set(FEATURES))

    if missing:
        return jsonify({
            "error": "Missing required features.",
            "missing_features": missing,
        }), 400

    if extra:
        return jsonify({
            "error": "Unexpected features supplied.",
            "unexpected_features": extra,
        }), 400

    for feature in FEATURES:
        value = payload[feature]

        if value is None:
            return jsonify({
                "error": f"Feature '{feature}' cannot be null."
            }), 400

        if feature in NUMERIC_FEATURES:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                return jsonify({
                    "error": f"Feature '{feature}' must be a finite number."
                }), 400
        elif not isinstance(value, str):
            return jsonify({
                "error": f"Feature '{feature}' must be a string."
            }), 400

    try:
        # Dict input preserves the feature names expected by the pipeline.
        import pandas as pd

        customer = pd.DataFrame(
            [[payload[feature] for feature in FEATURES]],
            columns=FEATURES,
        )

        probability = float(model.predict_proba(customer)[0, 1])
        prediction = int(probability >= THRESHOLD)

        return jsonify({
            "model_version": MODEL_VERSION,
            "prediction": "Churn" if prediction else "No churn",
            "churn_probability": round(probability, 4),
            "threshold": THRESHOLD,
        }), 200

    except Exception:
        app.logger.exception("Prediction failed.")
        return jsonify({
            "error": "Prediction failed."
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)