
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "ml" / "data" / "raw" / "Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "ml" / "artifacts" / "churn_model.joblib"
OUTPUT_PATH = ROOT / "ml" / "artifacts" / "evaluation_report.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
THRESHOLD = 0.50


def main():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError("Unexpected or missing target labels.")

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = joblib.load(MODEL_PATH)

    churn_probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (churn_probabilities >= THRESHOLD).astype(int)

    report = {
        "model_name": "churn_logistic_regression",
        "threshold": THRESHOLD,
        "test_samples": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, churn_probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No churn", "Churn"],
            output_dict=True,
            zero_division=0,
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("Evaluation completed.")
    print("Test samples:", report["test_samples"])
    print(f"Accuracy:  {report['accuracy']:.4f}")
    print(f"Precision: {report['precision']:.4f}")
    print(f"Recall:    {report['recall']:.4f}")
    print(f"ROC-AUC:   {report['roc_auc']:.4f}")
    print("Confusion matrix:", report["confusion_matrix"])
    print("Report saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
