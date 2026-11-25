
"""DeepShield: Verify the largest conflicting feature group."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = (
    PROJECT_ROOT
    / "data/raw/CIC-IDS2017/MachineLearningCSV/MachineLearningCVE"
)

REPORT_FILE = PROJECT_ROOT / "reports/label_conflicts.csv"

CHUNK_SIZE = 50_000


def verify_largest_conflict():

    report = pd.read_csv(REPORT_FILE)

    # Identify the largest conflicting feature group
    group_sizes = (
        report.groupby("feature_hash")["record_count"]
        .sum()
        .sort_values(ascending=False)
    )

    target_hash = str(group_sizes.index[0])
    expected_count = int(group_sizes.iloc[0])

    print("Largest conflicting group")
    print(f"Hash: {target_hash}")
    print(f"Expected records: {expected_count:,}")

    matching_records = []

    files = sorted(RAW_DIR.glob("*.csv"))

    for file in files:

        print(f"\nScanning: {file.name}")

        for chunk in pd.read_csv(file, chunksize=CHUNK_SIZE):

            chunk.columns = chunk.columns.str.strip()

            chunk["Label"] = (
                chunk["Label"].astype("string").str.strip()
            )

            features = chunk.drop(columns=["Label"])

            hashes = pd.util.hash_pandas_object(
                features,
                index=False
            ).astype("uint64")

            mask = hashes.astype(str).eq(target_hash)

            if mask.any():
                matching_records.append(chunk.loc[mask].copy())

    if not matching_records:
        print("No matching records found.")
        return

    matched = pd.concat(matching_records, ignore_index=True)

    feature_columns = [
        column for column in matched.columns
        if column != "Label"
    ]

    # Verify exact feature equality
    unique_features = matched[feature_columns].drop_duplicates()

    print("\n========== VERIFICATION REPORT ==========")
    print(f"Records found: {len(matched):,}")
    print(f"Expected records: {expected_count:,}")
    print(f"Unique feature combinations: {len(unique_features):,}")

    print("\n--- Label Distribution ---")
    print(matched["Label"].value_counts().to_string())

    if len(unique_features) == 1:
        print("\nCONFIRMED: All records have identical feature values.")
    else:
        print("\nWARNING: Multiple feature combinations found.")
        print(unique_features.head(5).to_string(index=False))

    print("==========================================")


if __name__ == "__main__":
    verify_largest_conflict()
