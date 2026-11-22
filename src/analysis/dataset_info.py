
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "CIC-IDS2017"
    / "MachineLearningCSV"
    / "MachineLearningCVE"
)

csv_files = sorted(DATA_DIR.glob("*.csv"))

print("=" * 45)
print("          DEEPSHIELD DATASET INFO")
print("=" * 45)

print(f"\nDataset directory: {DATA_DIR}")
print(f"Total CSV files: {len(csv_files)}")

total_size = 0

for file in csv_files:
    size_mb = file.stat().st_size / (1024 * 1024)
    total_size += file.stat().st_size

    print(f"{file.name}: {size_mb:.2f} MB")

print("-" * 45)
print(f"Total dataset size: {total_size / (1024 * 1024):.2f} MB")
print("=" * 45)
