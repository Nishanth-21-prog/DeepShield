
"""DeepShield: Investigate conflicting labels in CIC-IDS2017."""

from pathlib import Path
from collections import defaultdict

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = (
    PROJECT_ROOT
    / "data/raw/CIC-IDS2017/MachineLearningCSV/MachineLearningCVE"
)

OUTPUT_DIR = PROJECT_ROOT / "reports"
OUTPUT_FILE = OUTPUT_DIR / "label_conflicts.csv"

CHUNK_SIZE = 50_000


def analyze_label_conflicts():

    files = sorted(RAW_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError("Raw dataset files not found.")

    # Hash -> label counts
    conflict_groups = defaultdict(lambda: defaultdict(int))

    total_records = 0

    print("Starting label conflict analysis...\n")

    for file in files:

        print(f"Processing: {file.name}")

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

            total_records += len(chunk)

            for record_hash, label in zip(
                hashes.tolist(),
                chunk["Label"].tolist()
            ):
                conflict_groups[record_hash][str(label)] += 1

    rows = []

    for record_hash, label_counts in conflict_groups.items():

        if len(label_counts) > 1:

            for label, count in label_counts.items():

                rows.append({
                    "feature_hash": str(record_hash),
                    "label": label,
                    "record_count": count,
                    "total_group_records": sum(label_counts.values()),
                    "number_of_labels": len(label_counts)
                })

    report = pd.DataFrame(rows)

    if not report.empty:
        report = report.sort_values(
            ["total_group_records", "feature_hash"],
            ascending=[False, True]
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_FILE, index=False)

    print("\n========== LABEL CONFLICT REPORT ==========")
    print(f"Total records scanned: {total_records:,}")
    print(f"Conflicting feature groups: {report['feature_hash'].nunique() if not report.empty else 0:,}")
    print(f"Records in conflicting groups: {report.groupby('feature_hash')['record_count'].sum().sum() if not report.empty else 0:,}")
    print(f"Report saved to: {OUTPUT_FILE}")
    print("============================================")


if __name__ == "__main__":
    analyze_label_conflicts()
