
"""DeepShield: Summarize conflicting label combinations."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_FILE = PROJECT_ROOT / "reports/label_conflicts.csv"


def summarize_conflicts():

    if not REPORT_FILE.exists():
        raise FileNotFoundError("Label conflict report not found.")

    df = pd.read_csv(REPORT_FILE)

    print("========== CONFLICT SUMMARY ==========")

    # Count how many groups involve each label
    label_groups = (
        df.groupby("label")["feature_hash"]
        .nunique()
        .sort_values(ascending=False)
    )

    print("\n--- Labels Involved in Conflicts ---")
    print(label_groups.to_string())

    # Find the most frequent conflicting groups
    group_summary = (
        df.groupby("feature_hash")
        .agg(
            labels=("label", lambda x: " | ".join(sorted(x))),
            total_records=("record_count", "sum"),
            number_of_labels=("label", "nunique")
        )
        .reset_index()
        .sort_values("total_records", ascending=False)
    )

    print("\n--- Top 15 Conflicting Groups ---")
    print(
        group_summary[
            ["labels", "total_records", "number_of_labels"]
        ].head(15).to_string(index=False)
    )

    print("\n======================================")


if __name__ == "__main__":
    summarize_conflicts()
