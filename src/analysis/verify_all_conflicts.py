
"""DeepShield: Verify all conflicting feature groups."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = (
    PROJECT_ROOT
    / "data/raw/CIC-IDS2017/MachineLearningCSV/MachineLearningCVE"
)

REPORT_FILE = PROJECT_ROOT / "reports/label_conflicts.csv"
OUTPUT_FILE = PROJECT_ROOT / "reports/exact_label_conflicts.csv"

CHUNK_SIZE = 50_000


def verify_all_conflicts():

    if not REPORT_FILE.exists():
        raise FileNotFoundError("Conflict report not found.")

    report = pd.read_csv(
        REPORT_FILE,
        dtype={"feature_hash": str}
    )

    target_hashes = set(report["feature_hash"])

    matched_chunks = []
    total_scanned = 0

    files = sorted(RAW_DIR.glob("*.csv"))

    print("Starting verification of all conflicting groups...\n")

    for file in files:

        print(f"Scanning: {file.name}")

        for chunk in pd.read_csv(file, chunksize=CHUNK_SIZE):

            chunk.columns = chunk.columns.str.strip()
            chunk["Label"] = (
                chunk["Label"].astype("string").str.strip()
            )

            features = chunk.drop(columns=["Label"])

            hashes = pd.util.hash_pandas_object(
                features,
                index=False
            ).astype("uint64").astype(str)

            mask = hashes.isin(target_hashes)

            total_scanned += len(chunk)

            if mask.any():

                selected = chunk.loc[mask].copy()
                selected.insert(
                    0,
                    "feature_hash",
                    hashes.loc[mask].values
                )

                matched_chunks.append(selected)

    if not matched_chunks:
        print("No matching records found.")
        return

    matched = pd.concat(matched_chunks, ignore_index=True)

    feature_columns = [
        column for column in matched.columns
        if column not in ("Label", "feature_hash")
    ]

    # Group by the hash AND actual feature values.
    exact_groups = (
        matched.groupby(
            ["feature_hash"] + feature_columns,
            dropna=False,
            sort=False
        )
        .agg(
            total_records=("Label", "size"),
            number_of_labels=("Label", "nunique"),
            labels=(
                "Label",
                lambda values: " | ".join(
                    sorted(set(values.astype(str)))
                )
            )
        )
        .reset_index()
    )

    # Identify groups with identical features but different labels.
    conflicts = exact_groups[
        exact_groups["number_of_labels"] > 1
    ].copy()

    # Detect hashes that contain multiple distinct feature combinations.
    combinations_per_hash = exact_groups.groupby(
        "feature_hash"
    ).size()

    possible_hash_collisions = int(
        (combinations_per_hash > 1).sum()
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    conflicts.to_csv(OUTPUT_FILE, index=False)

    print("\n========== FULL VERIFICATION REPORT ==========")

    print(f"Records scanned: {total_scanned:,}")
    print(f"Candidate hash groups: {len(target_hashes):,}")
    print(f"Matching records found: {len(matched):,}")

    print(
        f"Exact feature groups verified: {len(exact_groups):,}"
    )

    print(
        f"Exact groups with conflicting labels: {len(conflicts):,}"
    )

    print(
        f"Records in exact conflicting groups: "
        f"{conflicts['total_records'].sum():,}"
    )

    print(
        f"Hashes with multiple feature combinations: "
        f"{possible_hash_collisions:,}"
    )

    print(f"\nReport saved to: {OUTPUT_FILE}")

    print("===============================================")


if __name__ == "__main__":
    verify_all_conflicts()
