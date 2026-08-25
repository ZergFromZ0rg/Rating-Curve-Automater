from pathlib import Path

from src.field_measurement_validation import clean_and_validate_measurements, read_measurement_excel


DATASET_PATH = Path(__file__).resolve().parent / "10_year_single_site_rating_curve_data.xlsx"
OUTPUT_PATH = Path(__file__).resolve().parent / "cleaned_measurements.csv"


def main() -> None:
    df = read_measurement_excel(DATASET_PATH, sheet_name="Measurements")
    cleaned = clean_and_validate_measurements(df)
    cleaned.to_csv(OUTPUT_PATH, index=False)

    valid_count = int(cleaned["is_valid"].sum())
    invalid_count = int((~cleaned["is_valid"]).sum())

    print(f"Using fixed dataset: {DATASET_PATH.name}")
    print(f"Read {len(cleaned)} rows from the default workbook")
    print(f"Valid rows: {valid_count}")
    print(f"Invalid rows: {invalid_count}")
    print(f"Cleaned output written to: {OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
