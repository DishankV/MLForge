
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "raw"
    / "Telco-Customer-Churn.csv"
)


def main():
    df = pd.read_csv(DATA_PATH)

    print("\n--- Dataset shape ---")
    print(df.shape)

    print("\n--- First five rows ---")
    print(df.head().to_string())

    print("\n--- Column names and data types ---")
    print(df.dtypes.to_string())

    print("\n--- Missing values per column ---")
    print(df.isna().sum().to_string())

    print("\n--- Target distribution ---")
    print(df["Churn"].value_counts(dropna=False).to_string())

    print("\n--- Target distribution (%) ---")
    print(
        (df["Churn"].value_counts(normalize=True) * 100)
        .round(2)
        .to_string()
    )

    print("\n--- Duplicate customer IDs ---")
    print(df["customerID"].duplicated().sum())

    print("\n--- Total charges data check ---")
    print(
        "Non-numeric TotalCharges:",
        pd.to_numeric(df["TotalCharges"], errors="coerce").isna().sum(),
    )


if __name__ == "__main__":
    main()
