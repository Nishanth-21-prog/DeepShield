
"""DeepShield: Analyze zero-duration network flows."""

from pathlib import Path
from collections import Counter

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data/processed/cleaned_dataset.csv"

CHUNK_SIZE = 50_000


def analyze_zero_duration():

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    total_rows = 0
    zero_duration = 0
    zero_with_both_missing = 0
    zero_with_other_pattern = 0

    labels = Counter()

    print("Starting zero-duration analysis...\n")

    for chunk in pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE):

        total_rows += len(chunk)

        duration = pd.to_numeric(
            chunk["Flow Duration"], errors="coerce"
        )

        bytes_missing = chunk["Flow Bytes/s"].isna()
        packets_missing = chunk["Flow Packets/s"].isna()

        zero_mask = duration.eq(0)

        zero_duration += int(zero_mask.sum())

        both_missing = zero_mask & bytes_missing & packets_missing

        zero_with_both_missing += int(both_missing.sum())

        zero_with_other_pattern += int(
            (zero_mask & ~(bytes_missing & packets_missing)).sum()
        )

        for label, count in chunk.loc[
            zero_mask, "Label"
        ].value_counts().items():
            labels[str(label)] += int(count)

    percentage = (
        zero_duration / total_rows * 100
        if total_rows else 0
    )

    print("\n========== ZERO-DURATION REPORT ==========")
    print(f"Total records: {total_rows:,}")
    print(f"Zero-duration records: {zero_duration:,}")
    print(f"Percentage of dataset: {percentage:.4f}%")
    print(f"Zero duration + both features missing: {zero_with_both_missing:,}")
    print(f"Zero duration with another missing-value pattern: {zero_with_other_pattern:,}")

    print("\n--- Zero-Duration Records By Label ---")

    for label, count in labels.most_common():
        print(f"{label}: {count:,}")

    print("===========================================")


if __name__ == "__main__":
    analyze_zero_duration()
