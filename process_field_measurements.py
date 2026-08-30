"""Convenience wrapper: clean/validate the bundled 10-year practice dataset.

Equivalent to::

    python3 -m src.field_measurement_validation --default-dataset
"""

from pathlib import Path

from src.field_measurement_validation import (
    DEFAULT_DATASET,
    DEFAULT_DATASET_SHEET,
    clean_measurements_to_csv,
)

OUTPUT_PATH = Path(__file__).resolve().parent / "cleaned_measurements.csv"


def main() -> None:
    print(f"Using fixed dataset: {DEFAULT_DATASET.name}")
    clean_measurements_to_csv(DEFAULT_DATASET, OUTPUT_PATH, DEFAULT_DATASET_SHEET)


if __name__ == "__main__":
    main()
