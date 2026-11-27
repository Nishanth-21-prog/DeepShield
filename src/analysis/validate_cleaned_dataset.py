import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
import sys

DATASET = Path("data/processed/cleaned_dataset.csv")
EXPECTED_RECORDS = 2_830_743
CHUNK_SIZE = 50_000

def main():
    if not DATASET.exists():
        print(f"Dataset not found: {DATASET}")
        sys.exit(1)

    total_records = 0
    total_missing = 0
    total_infinite = 0
    label_counts = Counter()

    print("Starting cleaned dataset validation...\n")

    for chunk in pd.read_csv(DATASET, chunksize=CHUNK_SIZE):
        total_records += len(chunk)
        total_missing += int(chunk.isna().sum().sum())

        numeric = chunk.select_dtypes(include=[np.number])
        total_infinite += int(np.isinf(numeric.to_numpy()).sum())

        label_col = next(
            (c for c in chunk.columns if c.strip().lower() == "label"),
            None
        )

        if label_col is None:
            print("ERROR: Label column not found.")
            sys.exit(1)

        label_counts.update(chunk[label_col].value_counts().to_dict())

        print(f"Validated {total_records:,} records", end="\r")

    print("\n\n========== VALIDATION REPORT ==========")
    print(f"Total records       : {total_records:,}")
    print(f"Missing values      : {total_missing:,}")
    print(f"Infinite values     : {total_infinite:,}")
    print(f"Expected records    : {EXPECTED_RECORDS:,}")

    print("\nClass distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"{label}: {count:,}")

    if total_records != EXPECTED_RECORDS or total_infinite != 0:
        print("\nVALIDATION FAILED")
        sys.exit(1)

    print("\nVALIDATION PASSED")

if __name__ == "__main__":
    main()
