
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------- Paths and reproducibility ----------

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "ml" / "data" / "raw" / "Telco-Customer-Churn.csv"
)
ARTIFACT_DIR = ROOT / "ml" / "artifacts"
RANDOM_STATE = 42


def main():
    # 1. Load and clean the data
    df = pd.read_csv(DATA_PATH)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    # Customer ID is an identifier, not a predictive feature.
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError("Unexpected or missing target labels.")

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    print("Rows:", len(df))
    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)
    print("Missing TotalCharges:", X["TotalCharges"].isna().sum())

    # 2. Split before fitting preprocessing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 3. Define preprocessing separately for each feature type
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    # 4. Keep preprocessing and model together
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        )),
    ])

    # 5. Train only on training data
    model.fit(X_train, y_train)

    # 6. Evaluate on the held-out test data
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "logistic_regression",
        "random_state": RANDOM_STATE,
        "test_size": 0.20,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No churn", "Churn"],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
    }

    # 7. Compare against a simple majority-class baseline
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)

    metrics["baseline_accuracy"] = float(
        accuracy_score(y_test, baseline_predictions)
    )

    print("\n--- Evaluation ---")
    print("Baseline accuracy:", round(metrics["baseline_accuracy"], 4))
    print("Model accuracy:", round(metrics["accuracy"], 4))
    print("ROC-AUC:", round(metrics["roc_auc"], 4))
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["No churn", "Churn"],
        zero_division=0,
    ))
    print("Confusion matrix [actual rows, predicted columns]:")
    print(metrics["confusion_matrix"])

    # 8. Save the full pipeline and evaluation report
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, ARTIFACT_DIR / "churn_model.joblib")

    with (ARTIFACT_DIR / "metrics.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved model:", ARTIFACT_DIR / "churn_model.joblib")
    print("Saved metrics:", ARTIFACT_DIR / "metrics.json")


if __name__ == "__main__":
    main()
