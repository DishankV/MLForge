
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil
import sys

import sklearn

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "ml" / "artifacts"
REGISTRY_DIR = ROOT / "ml" / "registry" / "models"

MODEL_VERSION = "churn-logreg-v1"

MODEL_SOURCE = ARTIFACT_DIR / "churn_model.joblib"
METRICS_SOURCE = ARTIFACT_DIR / "metrics.json"
EVALUATION_SOURCE = ARTIFACT_DIR / "evaluation_report.json"

VERSION_DIR = REGISTRY_DIR / MODEL_VERSION


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    required_files = [
        MODEL_SOURCE,
        METRICS_SOURCE,
        EVALUATION_SOURCE,
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required artifact not found: {path}"
            )

    if VERSION_DIR.exists():
        raise FileExistsError(
            f"Model version already exists: {VERSION_DIR}. "
            "Choose a new version rather than overwriting it."
        )

    VERSION_DIR.mkdir(parents=True)

    try:
        model_destination = VERSION_DIR / "model.joblib"
        metrics_destination = VERSION_DIR / "metrics.json"
        evaluation_destination = VERSION_DIR / "evaluation_report.json"

        shutil.copy2(MODEL_SOURCE, model_destination)
        shutil.copy2(METRICS_SOURCE, metrics_destination)
        shutil.copy2(EVALUATION_SOURCE, evaluation_destination)

        with evaluation_destination.open(
            "r", encoding="utf-8"
        ) as file:
            evaluation = json.load(file)

        metadata = {
            "model_name": "telco_churn_classifier",
            "model_version": MODEL_VERSION,
            "model_type": "LogisticRegression",
            "framework": "scikit-learn",
            "framework_version": sklearn.__version__,
            "python_version": sys.version.split()[0],
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "status": "candidate",
            "artifact_path": str(
                model_destination.relative_to(ROOT)
            ),
            "metrics_path": str(
                metrics_destination.relative_to(ROOT)
            ),
            "evaluation_report_path": str(
                evaluation_destination.relative_to(ROOT)
            ),
            "artifact_sha256": sha256_file(model_destination),
            "evaluation_summary": {
                "accuracy": evaluation["accuracy"],
                "precision": evaluation["precision"],
                "recall": evaluation["recall"],
                "roc_auc": evaluation["roc_auc"],
            },
        }

        metadata_path = VERSION_DIR / "metadata.json"

        with metadata_path.open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(metadata, file, indent=2)

    except Exception:
        shutil.rmtree(VERSION_DIR, ignore_errors=True)
        raise

    print("Model registered successfully.")
    print("Version:", metadata["model_version"])
    print("Status:", metadata["status"])
    print("Accuracy:", metadata["evaluation_summary"]["accuracy"])
    print("ROC-AUC:", metadata["evaluation_summary"]["roc_auc"])
    print("Artifact SHA-256:", metadata["artifact_sha256"])
    print("Registry entry:", VERSION_DIR)


if __name__ == "__main__":
    main()
