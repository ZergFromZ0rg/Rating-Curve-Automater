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
        bad_quality = result[quality_col].map(_parse_quality).isin({"bad", "poor", "unreliable", "reject", "rejected"})
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
        if quality_col is not None and _parse_quality(row[quality_col]) in {"bad", "poor", "unreliable", "reject", "rejected"}:
            reasons.append("quality flag indicates poor data")
        return "; ".join(reasons)

    result["validation_notes"] = result.apply(note_for_row, axis=1)
    result.loc[~result["is_valid"], "validation_notes"] = result.loc[~result["is_valid"], :].apply(note_for_row, axis=1)

    return result


def read_measurement_excel(input_path: str | Path, sheet_name: str | int | None = None) -> pd.DataFrame:
    path = Path(input_path)
    excel = pd.ExcelFile(path)
    sheet_to_read = sheet_name if sheet_name is not None else excel.sheet_names[0]
    df = pd.read_excel(path, sheet_name=sheet_to_read)
    return df


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Clean and validate field measurements for rating curve analysis.")
    parser.add_argument("input_excel", type=str, help="Path to the Excel file containing field measurements.")
    parser.add_argument("--sheet-name", type=str, default=None, help="Optional Excel sheet name to read.")
    parser.add_argument("--output-csv", type=str, default="cleaned_measurements.csv", help="CSV output file to write.")
    args = parser.parse_args()

    df = read_measurement_excel(args.input_excel, args.sheet_name)
    cleaned = clean_and_validate_measurements(df)
    cleaned.to_csv(args.output_csv, index=False)
    valid_count = int(cleaned["is_valid"].sum())
    invalid_count = int((~cleaned["is_valid"]).sum())
    print(f"Read {len(cleaned)} rows from {args.input_excel}")
    print(f"Valid rows: {valid_count}")
    print(f"Invalid rows: {invalid_count}")
    print(f"Cleaned output written to: {args.output_csv}")


if __name__ == "__main__":
    main()
