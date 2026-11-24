
"""DeepShield: Analyze duplicate network-flow records."""

from pathlib import Path
from collections import Counter, defaultdict
import hashlib

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = (
    PROJECT_ROOT
    / "data/raw/CIC-IDS2017/MachineLearningCSV/MachineLearningCVE"
)

CHUNK_SIZE = 50_000


def create_hashes(chunk):
    """Create hashes using all features, excluding Label."""

    features = chunk.drop(columns=["Label"])

    return pd.util.hash_pandas_object(
        features,
        index=False
    ).astype("uint64")


def analyze_duplicates():

    files = sorted(RAW_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError("Raw CSV files not found.")

    hash_counts = Counter()
    hash_labels = defaultdict(set)

    total_records = 0

    print("Starting duplicate analysis...\n")

    for file in files:

        print(f"Processing: {file.name}")

        for chunk in pd.read_csv(file, chunksize=CHUNK_SIZE):

            chunk.columns = chunk.columns.str.strip()

            chunk["Label"] = (
                chunk["Label"].astype("string").str.strip()
            )

            hashes = create_hashes(chunk)

            total_records += len(chunk)

            hash_counts.update(hashes.tolist())

            for record_hash, label in zip(
                hashes.tolist(),
                chunk["Label"].tolist()
            ):
                hash_labels[record_hash].add(str(label))

    repeated_groups = {
        key: count
        for key, count in hash_counts.items()
        if count > 1
    }

    repeated_records = sum(
        count - 1 for count in repeated_groups.values()
    )

    conflicting_groups = {
        key: count
        for key, count in repeated_groups.items()
        if len(hash_labels[key]) > 1
    }

    conflicting_records = sum(
        hash_counts[key] for key in conflicting_groups
    )

    print("\n========== DUPLICATE REPORT ==========")

    print(f"Total records: {total_records:,}")
    print(f"Unique feature combinations: {len(hash_counts):,}")
    print(f"Repeated feature groups: {len(repeated_groups):,}")
    print(f"Repeated records beyond first occurrence: {repeated_records:,}")

    print("\n--- Conflicting Labels ---")

    print(f"Conflicting feature groups: {len(conflicting_groups):,}")
    print(f"Records in conflicting groups: {conflicting_records:,}")

    print("\n=======================================")


if __name__ == "__main__":
    analyze_duplicates()
