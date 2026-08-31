from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.cleaning import clean_numeric_series, coerce_datetime, drop_footer_rows
from src.schema import DATE, DISCHARGE_CMS, NOTES, QUALITY, SITE, STAGE_M, TIME, ensure_canonical

REJECT_QUALITY_VALUES = {"bad", "poor", "unreliable", "reject", "rejected"}
#: An identical stage value repeated this many times (per site) is treated as a
#: placeholder / gauge fault, not a real measurement.
REPEATED_STAGE_MIN = 3
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


def _stuck_stage_mask(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Flag rows whose stage value repeats >= REPEATED_STAGE_MIN times per site."""
    mask = pd.Series(False, index=frame.index)
    count = pd.Series(0, index=frame.index)
    site_key = frame[SITE].astype(str) if SITE in frame.columns else pd.Series("", index=frame.index)
    for _, idx in frame.groupby([site_key, frame[STAGE_M]], dropna=True).groups.items():
        if len(idx) >= REPEATED_STAGE_MIN:
            mask.loc[idx] = True
            count.loc[idx] = len(idx)
    return mask, count


def clean_and_validate_measurements(
    df: pd.DataFrame,
    column_overrides: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Clean and validate a field measurement table for rating curve work.

    ``df`` may use the canonical schema (see :mod:`src.schema`) or any workbook
    headers that :func:`~src.schema.resolve_columns` can map. The returned frame
    always has canonical columns plus ``is_valid`` / ``validation_notes`` /
    ``has_warning`` / ``warning_notes``.
    """
    result = ensure_canonical(df, overrides=column_overrides).copy()

    has_quality = QUALITY in result.columns
    has_notes = NOTES in result.columns

    # Safety net for callers that bypass load_measurements(): drop footer rows
    # and coerce messy values here too. All of this is idempotent on data that
    # the loader already cleaned.
    result, _ = drop_footer_rows(result, [DATE, SITE, NOTES])

    time_series = result[TIME] if TIME in result.columns else None
    result[DATE] = coerce_datetime(result[DATE], time_series)
    if time_series is not None:
        result = result.drop(columns=[TIME])

    stage_values, _ = clean_numeric_series(result[STAGE_M])
    discharge_values, discharge_censored = clean_numeric_series(result[DISCHARGE_CMS])
    result[STAGE_M] = stage_values
    result[DISCHARGE_CMS] = discharge_values

    result["is_valid"] = True
    result["validation_notes"] = ""

    invalid_mask = pd.Series(False, index=result.index)

    invalid_mask |= result[DATE].isna()
    invalid_mask |= result[STAGE_M].isna()
    invalid_mask |= result[DISCHARGE_CMS].isna()

    invalid_mask |= result[STAGE_M] <= 0
    invalid_mask |= result[DISCHARGE_CMS] < 0

    if has_quality:
        bad_quality = result[QUALITY].map(_parse_quality).isin(REJECT_QUALITY_VALUES)
        invalid_mask |= bad_quality

    stuck_mask, stuck_count = _stuck_stage_mask(result)
    invalid_mask |= stuck_mask

    result.loc[invalid_mask, "is_valid"] = False

    def note_for_row(row: pd.Series) -> str:
        reasons = []
        if pd.isna(row[DATE]):
            reasons.append("invalid date")
        if pd.isna(row[STAGE_M]) or row[STAGE_M] <= 0:
            reasons.append("stage missing or <= 0")
        if pd.isna(row[DISCHARGE_CMS]) or row[DISCHARGE_CMS] < 0:
            reasons.append("discharge missing or negative")
        if has_quality and _parse_quality(row[QUALITY]) in REJECT_QUALITY_VALUES:
            reasons.append("quality flag indicates poor data")
        if stuck_mask.get(row.name, False):
            reasons.append(
                f"stage {row[STAGE_M]:g} repeats {int(stuck_count[row.name])}x "
                f"(likely placeholder / gauge fault)"
            )
        return "; ".join(reasons)

    def warning_for_row(row: pd.Series) -> str:
        reasons = []
        if has_quality:
            quality_text = _parse_quality(row[QUALITY])
            if quality_text in WARN_QUALITY_VALUES:
                reasons.append(f"quality flagged '{str(row[QUALITY]).strip()}'")
        if has_notes and not pd.isna(row[NOTES]):
            note_text = str(row[NOTES]).strip()
            if any(keyword in note_text.lower() for keyword in WARN_NOTE_KEYWORDS):
                reasons.append(f"field note: {note_text}")
        if bool(discharge_censored.get(row.name, False)):
            reasons.append("censored discharge value taken at face value")
        return "; ".join(reasons)

    result["validation_notes"] = result.apply(note_for_row, axis=1)

    result["warning_notes"] = result.apply(warning_for_row, axis=1)
    # Warnings only apply to rows that are otherwise usable in the fit.
    result.loc[~result["is_valid"], "warning_notes"] = ""
    result["has_warning"] = result["warning_notes"].str.len() > 0

    return result


def read_measurement_excel(input_path: str | Path, sheet_name: str | int | None = None) -> pd.DataFrame:
    """Load a workbook into a canonical frame (kept for backwards compatibility)."""
    from src.loader import load_measurements

    canonical, _ = load_measurements(input_path, sheet=sheet_name)
    return canonical


DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "10_year_single_site_rating_curve_data.xlsx"
DEFAULT_DATASET_SHEET = "Measurements"


def clean_measurements_to_csv(
    input_excel: str | Path,
    output_csv: str | Path,
    sheet_name: str | int | None = None,
) -> pd.DataFrame:
    """Load a workbook, clean/validate it, and write the result to CSV.

    Returns the cleaned dataframe and prints a short summary.
    """
    from src.loader import load_measurements

    canonical, report = load_measurements(input_excel, sheet=sheet_name)
    cleaned = clean_and_validate_measurements(canonical)
    cleaned.to_csv(output_csv, index=False)

    print(report.describe())
    print(f"Valid rows: {int(cleaned['is_valid'].sum())}")
    print(f"Invalid rows: {int((~cleaned['is_valid']).sum())}")
    print(f"Valid rows with warnings: {int(cleaned['has_warning'].sum())}")
    print(f"Cleaned output written to: {Path(output_csv).name}")
    return cleaned


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Clean and validate field measurements for rating curve analysis.")
    parser.add_argument("input_excel", type=str, nargs="?", default=None, help="Path to the Excel/CSV file containing field measurements.")
    parser.add_argument("--default-dataset", action="store_true", help=f"Use the bundled dataset ({DEFAULT_DATASET.name}).")
    parser.add_argument("--sheet-name", type=str, default=None, help="Excel sheet name (default: auto-detect).")
    parser.add_argument("--output-csv", type=str, default="cleaned_measurements.csv", help="CSV output file to write.")
    args = parser.parse_args()

    if args.default_dataset:
        input_excel: str | Path = DEFAULT_DATASET
        sheet_name = args.sheet_name
    elif args.input_excel:
        input_excel = args.input_excel
        sheet_name = args.sheet_name
    else:
        parser.error("provide an input_excel path or --default-dataset")

    clean_measurements_to_csv(input_excel, args.output_csv, sheet_name)


if __name__ == "__main__":
    main()
