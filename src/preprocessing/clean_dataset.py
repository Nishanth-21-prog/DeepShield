
"""Stage 1: Safely clean CIC-IDS2017 CSV files."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = (
    PROJECT_ROOT
    / "data/raw/CIC-IDS2017/MachineLearningCSV/MachineLearningCVE"
)

PROCESSED_DATA_DIR = PROJECT_ROOT / "data/processed"

OUTPUT_FILE = PROCESSED_DATA_DIR / "cleaned_dataset.csv"
TEMP_FILE = PROCESSED_DATA_DIR / "cleaned_dataset.tmp"

CHUNK_SIZE = 50_000


def clean_dataset():

    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError("No CIC-IDS2017 CSV files found.")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    total_infinities = 0
    total_missing = 0
    first_chunk = True

    print("Starting DeepShield Stage 1 cleaning...\n")

    try:
        for csv_file in csv_files:

            print(f"Processing: {csv_file.name}")

            for chunk in pd.read_csv(
                csv_file,
                chunksize=CHUNK_SIZE
            ):

                chunk.columns = chunk.columns.str.strip()

                if "Label" not in chunk.columns:
                    raise ValueError("Label column missing.")

                chunk["Label"] = (
                    chunk["Label"]
                    .astype("string")
                    .str.strip()
                )

                feature_columns = [
                    col for col in chunk.columns
                    if col != "Label"
                ]

                chunk[feature_columns] = chunk[
                    feature_columns
                ].apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                numeric = chunk[feature_columns]

                total_infinities += int(
                    np.isinf(
                        numeric.to_numpy(
                            dtype=float,
                            na_value=np.nan
                        )
                    ).sum()
                )

                chunk[feature_columns] = numeric.replace(
                    [np.inf, -np.inf],
                    np.nan
                )

                total_missing += int(
                    chunk.isna().sum().sum()
                )

                chunk.to_csv(
                    TEMP_FILE,
                    mode="w" if first_chunk else "a",
                    header=first_chunk,
                    index=False
                )

                first_chunk = False
                total_rows += len(chunk)

        if first_chunk:
            raise RuntimeError("No records processed.")

        TEMP_FILE.replace(OUTPUT_FILE)

    except Exception:
        TEMP_FILE.unlink(missing_ok=True)
        raise

    print("\n========== STAGE 1 SUMMARY ==========")
    print(f"Files processed: {len(csv_files)}")
    print(f"Total records preserved: {total_rows:,}")
    print(f"Infinite values converted: {total_infinities:,}")
    print(f"Missing values remaining: {total_missing:,}")
    print(f"Output: {OUTPUT_FILE}")
    print("Duplicates removed: 0")
    print("=====================================")


if __name__ == "__main__":
    clean_dataset()
