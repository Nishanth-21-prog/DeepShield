import pandas as pd
from pathlib import Path

DATASET = Path("data/processed/cleaned_dataset.csv")
CHUNK_SIZE = 50_000

def main():
    if not DATASET.exists():
        raise FileNotFoundError(DATASET)

    minimum = {}
    maximum = {}
    missing = {}
    valid = {}

    print("Starting feature range analysis...\n")

    for chunk in pd.read_csv(DATASET, chunksize=CHUNK_SIZE):
        features = [
            col for col in chunk.columns
            if col.strip().lower() != "label"
        ]

        for col in features:
            values = pd.to_numeric(chunk[col], errors="coerce")

            low = values.min()
            high = values.max()

            missing[col] = missing.get(col, 0) + int(values.isna().sum())
            valid[col] = valid.get(col, 0) + int(values.notna().sum())

            if pd.notna(low):
                minimum[col] = min(minimum.get(col, low), low)

            if pd.notna(high):
                maximum[col] = max(maximum.get(col, high), high)

        print("Processed records:", f"{sum(valid.values()) // len(features):,}", end="\r")

    print("\n\n========== FEATURE ANALYSIS ==========")

    constant = []

    for col in minimum:
        if minimum[col] == maximum[col]:
            constant.append(col)

        print(f"\nFeature: {col}")
        print(f"  Minimum : {minimum[col]}")
        print(f"  Maximum : {maximum[col]}")
        print(f"  Missing : {missing[col]:,}")

    print("\n========== SUMMARY ==========")
    print("Total features:", len(minimum))
    print("Constant features:", len(constant))

    if constant:
        print("\nConstant feature candidates:")
        for col in constant:
            print("-", col)

    print("\nFeature range analysis completed.")

if __name__ == "__main__":
    main()
