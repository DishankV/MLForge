
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_DIR = ROOT / "ml" / "registry" / "models"

MODEL_VERSION = "churn-logreg-v1"
VERSION_DIR = REGISTRY_DIR / MODEL_VERSION
METADATA_PATH = VERSION_DIR / "metadata.json"
EVALUATION_PATH = VERSION_DIR / "evaluation_report.json"
MODEL_PATH = VERSION_DIR / "model.joblib"

MIN_ROC_AUC = 0.80
MIN_RECALL = 0.55


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    required_files = [
        METADATA_PATH,
        EVALUATION_PATH,
        MODEL_PATH,
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required registry file not found: {path}"
            )

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    with EVALUATION_PATH.open("r", encoding="utf-8") as file:
        evaluation = json.load(file)

    actual_hash = sha256_file(MODEL_PATH)
    expected_hash = metadata.get("artifact_sha256")

    checks = {
        "roc_auc": evaluation.get("roc_auc", 0) >= MIN_ROC_AUC,
        "recall": evaluation.get("recall", 0) >= MIN_RECALL,
        "artifact_integrity": (
            expected_hash is not None
            and actual_hash == expected_hash
        ),
    }

    print(f"Model version: {MODEL_VERSION}")
    print("\nPromotion checks:")

    for check_name, passed in checks.items():
        print(f"  {check_name}: {'PASS' if passed else 'FAIL'}")

    if all(checks.values()):
        metadata["status"] = "approved"
        metadata["promotion_checks"] = {
            "minimum_roc_auc": MIN_ROC_AUC,
            "minimum_recall": MIN_RECALL,
            "results": checks,
        }
        print("\nResult: APPROVED")
    else:
        metadata["status"] = "candidate"
        metadata["promotion_checks"] = {
            "minimum_roc_auc": MIN_ROC_AUC,
            "minimum_recall": MIN_RECALL,
            "results": checks,
        }
        print("\nResult: NOT APPROVED")

    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("Updated metadata:", METADATA_PATH)
    print("Final status:", metadata["status"])


if __name__ == "__main__":
    main()
