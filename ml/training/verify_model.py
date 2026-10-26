
from pathlib import Path
import json
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "ml" / "artifacts"

model_path = ARTIFACT_DIR / "churn_model.joblib"
metrics_path = ARTIFACT_DIR / "metrics.json"

model = joblib.load(model_path)

with metrics_path.open("r", encoding="utf-8") as file:
    metrics = json.load(file)

sample = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 65.0,
    "TotalCharges": 780.0,
}])

prediction = int(model.predict(sample)[0])
probability = float(model.predict_proba(sample)[0, 1])

print("Model loaded:", model_path.exists())
print("Metrics loaded:", metrics_path.exists())
print("Recorded ROC-AUC:", metrics["roc_auc"])
print("Prediction:", "Churn" if prediction == 1 else "No churn")
print(f"Churn probability: {probability:.4f}")
