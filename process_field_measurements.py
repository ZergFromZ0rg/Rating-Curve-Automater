"""Convenience wrapper: clean/validate the bundled 10-year practice dataset,
writing ``cleaned_measurements.csv`` next to this file.

Equivalent to ``rca validate --default-dataset --output-csv cleaned_measurements.csv``.
"""

from pathlib import Path

from rating_curve_automater.field_measurement_validation import (
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
