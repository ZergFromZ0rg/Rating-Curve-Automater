from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def _normalize_column_name(name: object) -> str:
    if pd.isna(name):
        return ""
    text = str(name).strip().lower()
    text = text.replace("(m³/s)", "")
    text = text.replace("(m3/s)", "")
    text = text.replace("(masl)", "")
    text = text.replace("(m)", "")
    text = text.replace("-", " ")
    text = " ".join(text.split())
    return text


def _find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {_normalize_column_name(col): col for col in columns}
    for candidate in candidates:
        if _normalize_column_name(candidate) in normalized:
            return normalized[_normalize_column_name(candidate)]
    return None


REJECT_QUALITY_VALUES = {"bad", "poor", "unreliable", "reject", "rejected"}
WARN_QUALITY_VALUES = {"fair", "questionable", "estimated", "provisional"}
WARN_NOTE_KEYWORDS = (
    "backwater",
    "vegetation",
    "shallow",
    "low velocity",
    "ice",
    "debris",
    "overbank",
    "unsteady",
)


def _parse_quality(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def clean_and_validate_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate a field measurement table for rating curve work.

    Supports common hydrology column names such as:
    - Date
    - Stage Above Bed (m)
    - Water Surface Elevation (masl)
    - Measured Discharge Q (m³/s)
    - Quality
    - Field Notes
    """
    result = df.copy()

    date_col = _find_column(result.columns, [
        "Date",
        "Measurement Date",
        "date",
        "datetime",
    ])
    stage_col = _find_column(result.columns, [
        "Stage Above Bed",
        "Stage Above Bed (m)",
        "Stage",
        "SG",
        "Stage m",
        "Water level",
        "Water Surface Elevation",
        "Water Surface Elevation (masl)",
        "Elev",
        "Elev (m)",
    ])
    discharge_col = _find_column(result.columns, [
        "Measured Discharge Q",
        "Measured Discharge Q (m3/s)",
        "Measured Discharge Q (m³/s)",
        "Discharge",
        "Flow",
        "Flow (m3/s)",
        "Q",
        "Q (m3/s)",
    ])
    quality_col = _find_column(result.columns, [
        "Quality",
        "Field Quality",
        "quality",
    ])
    notes_col = _find_column(result.columns, [
        "Field Notes",
        "Notes",
        "Comment",
        "Comments",
        "Remarks",
    ])

    if date_col is None:
        raise ValueError("Could not find a date column in the Excel sheet.")
    if stage_col is None:
        raise ValueError("Could not find a stage column in the Excel sheet.")
    if discharge_col is None:
        raise ValueError("Could not find a discharge column in the Excel sheet.")

    result[date_col] = pd.to_datetime(result[date_col], errors="coerce")
    result[stage_col] = pd.to_numeric(result[stage_col], errors="coerce")
    result[discharge_col] = pd.to_numeric(result[discharge_col], errors="coerce")

    result["is_valid"] = True
    result["validation_notes"] = ""

    invalid_mask = pd.Series(False, index=result.index)

    invalid_mask |= result[date_col].isna()
    invalid_mask |= result[stage_col].isna()
    invalid_mask |= result[discharge_col].isna()

    stage_bad = result[stage_col] <= 0
    discharge_bad = result[discharge_col] < 0
    invalid_mask |= stage_bad
    invalid_mask |= discharge_bad

    if quality_col is not None:
        bad_quality = result[quality_col].map(_parse_quality).isin(REJECT_QUALITY_VALUES)
        invalid_mask |= bad_quality

    result.loc[invalid_mask, "is_valid"] = False

    def note_for_row(row: pd.Series) -> str:
        reasons = []
        if pd.isna(row[date_col]):
            reasons.append("invalid date")
        if pd.isna(row[stage_col]) or row[stage_col] <= 0:
            reasons.append("stage missing or <= 0")
        if pd.isna(row[discharge_col]) or row[discharge_col] < 0:
            reasons.append("discharge missing or negative")
        if quality_col is not None and _parse_quality(row[quality_col]) in REJECT_QUALITY_VALUES:
            reasons.append("quality flag indicates poor data")
        return "; ".join(reasons)

    def warning_for_row(row: pd.Series) -> str:
        reasons = []
        if quality_col is not None:
            quality_text = _parse_quality(row[quality_col])
            if quality_text in WARN_QUALITY_VALUES:
                reasons.append(f"quality flagged '{str(row[quality_col]).strip()}'")
        if notes_col is not None and not pd.isna(row[notes_col]):
            note_text = str(row[notes_col]).strip()
            if any(keyword in note_text.lower() for keyword in WARN_NOTE_KEYWORDS):
                reasons.append(f"field note: {note_text}")
        return "; ".join(reasons)

    result["validation_notes"] = result.apply(note_for_row, axis=1)

    result["warning_notes"] = result.apply(warning_for_row, axis=1)
    # Warnings only apply to rows that are otherwise usable in the fit.
    result.loc[~result["is_valid"], "warning_notes"] = ""
    result["has_warning"] = result["warning_notes"].str.len() > 0

    return result


def read_measurement_excel(input_path: str | Path, sheet_name: str | int | None = None) -> pd.DataFrame:
    path = Path(input_path)
    excel = pd.ExcelFile(path)
    sheet_to_read = sheet_name if sheet_name is not None else excel.sheet_names[0]
    df = pd.read_excel(path, sheet_name=sheet_to_read)
    return df


DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "10_year_single_site_rating_curve_data.xlsx"
DEFAULT_DATASET_SHEET = "Measurements"


def clean_measurements_to_csv(
    input_excel: str | Path,
    output_csv: str | Path,
    sheet_name: str | int | None = None,
) -> pd.DataFrame:
    """Read an Excel workbook, clean/validate it, and write the result to CSV.

    Returns the cleaned dataframe and prints a short summary.
    """
    df = read_measurement_excel(input_excel, sheet_name)
    cleaned = clean_and_validate_measurements(df)
    cleaned.to_csv(output_csv, index=False)

    print(f"Read {len(cleaned)} rows from {Path(input_excel).name}")
    print(f"Valid rows: {int(cleaned['is_valid'].sum())}")
    print(f"Invalid rows: {int((~cleaned['is_valid']).sum())}")
    print(f"Valid rows with warnings: {int(cleaned['has_warning'].sum())}")
    print(f"Cleaned output written to: {Path(output_csv).name}")
    return cleaned


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Clean and validate field measurements for rating curve analysis.")
    parser.add_argument("input_excel", type=str, nargs="?", default=None, help="Path to the Excel file containing field measurements.")
    parser.add_argument("--default-dataset", action="store_true", help=f"Use the bundled dataset ({DEFAULT_DATASET.name}).")
    parser.add_argument("--sheet-name", type=str, default=None, help="Optional Excel sheet name to read.")
    parser.add_argument("--output-csv", type=str, default="cleaned_measurements.csv", help="CSV output file to write.")
    args = parser.parse_args()

    if args.default_dataset:
        input_excel = DEFAULT_DATASET
        sheet_name = args.sheet_name or DEFAULT_DATASET_SHEET
    elif args.input_excel:
        input_excel = args.input_excel
        sheet_name = args.sheet_name
    else:
        parser.error("provide an input_excel path or --default-dataset")

    clean_measurements_to_csv(input_excel, args.output_csv, sheet_name)


if __name__ == "__main__":
    main()
