
"""DeepShield: Analyze missing values in the cleaned CIC-IDS2017 dataset."""

from pathlib import Path
from collections import Counter

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = PROJECT_ROOT / "data/processed/cleaned_dataset.csv"

CHUNK_SIZE = 50_000


def analyze_missing_values():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_FILE}"
        )

    feature_missing = Counter()
    label_missing = Counter()

    total_rows = 0
    total_missing = 0

    print("Starting missing value analysis...\n")

    for chunk in pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE):

        total_rows += len(chunk)

        missing = chunk.isna()

        total_missing += int(missing.sum().sum())

        # Count missing values in each feature
        feature_counts = missing.sum()

        for column, count in feature_counts.items():
            if column != "Label" and count > 0:
                feature_missing[column] += int(count)

        # Identify labels associated with missing values
        rows_with_missing = missing.any(axis=1)

        affected_rows = chunk.loc[rows_with_missing]

        for label, count in affected_rows["Label"].value_counts().items():
            label_missing[str(label)] += int(count)

    print("\n========== MISSING VALUE REPORT ==========")

    print(f"Total records scanned: {total_rows:,}")
    print(f"Total missing values: {total_missing:,}")

    print("\n--- Missing Values By Feature ---")

    if feature_missing:
        for feature, count in feature_missing.most_common():
            print(f"{feature}: {count:,}")
    else:
        print("No missing values found in features.")

    print("\n--- Records With Missing Values By Label ---")

    if label_missing:
        for label, count in label_missing.most_common():
            print(f"{label}: {count:,}")
    else:
        print("No affected records found.")

    print("\n==========================================")


if __name__ == "__main__":
    analyze_missing_values()
