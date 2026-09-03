from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rating_curve_automater.cleaning import (
    clean_numeric_series,
    clean_uncertainty_series,
    coerce_datetime,
    drop_footer_rows,
)
from rating_curve_automater.schema import (
    DATE,
    DISCHARGE_CMS,
    DISCHARGE_UNCERTAINTY,
    NOTES,
    QUALITY,
    SITE,
    STAGE_M,
    TIME,
    ensure_canonical,
)

REJECT_QUALITY_VALUES = {"bad", "poor", "unreliable", "reject", "rejected"}
#: How many identical stage readings (per site) it takes before the run is even
#: considered for a placeholder / stuck-gauge flag. A run this long is only
#: rejected when it *also* looks non-physical (see :func:`_stuck_stage_mask`):
#: the discharge repeats too, or the readings are consecutive field visits. A
#: stage value that simply recurs across the record with varying discharge (a
#: common low-flow stage) is left alone.
REPEATED_STAGE_MIN = 3
#: Discharge values within this relative spread count as "the same" for the
#: duplicated-row test.
_STUCK_Q_REL_TOL = 0.02
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
    """Flag rows that look like a placeholder stage or a stuck gauge, per site.

    A group of ``>= REPEATED_STAGE_MIN`` rows sharing one stage value is rejected
    only when it is also non-physical:

    * **discharge repeats too** — identical stage *and* discharge is a duplicated
      / placeholder row, not a measurement; or
    * **consecutive field visits** — the same reading across an unbroken run of
      visits (date order) is a gauge stuck at one value.

    A stage value that merely recurs across the record with genuinely different
    discharges (a common low-flow stage, revisited years apart) is left valid.

    Returns ``(mask, detail)`` where ``detail`` carries the per-row reason text.
    """
    mask = pd.Series(False, index=frame.index)
    detail = pd.Series("", index=frame.index, dtype="object")
    site_key = frame[SITE].astype(str) if SITE in frame.columns else pd.Series("", index=frame.index)

    # Rank each row among its own site's visits so a "consecutive run" test is
    # not fooled by two sites' dates interleaving.
    if DATE in frame.columns and frame[DATE].notna().any():
        visit_order = frame[DATE].groupby(site_key).rank(method="first", na_option="bottom")
    else:
        visit_order = frame.groupby(site_key).cumcount().astype(float)

    discharge = (
        pd.to_numeric(frame[DISCHARGE_CMS], errors="coerce")
        if DISCHARGE_CMS in frame.columns
        else None
    )

    for key, idx in frame.groupby([site_key, frame[STAGE_M]], dropna=True).groups.items():
        stage_val = key[1] if isinstance(key, tuple) else key
        if pd.isna(stage_val) or len(idx) < REPEATED_STAGE_MIN:
            continue

        q_repeats = False
        if discharge is not None:
            q = discharge.loc[idx].dropna()
            if len(q) >= REPEATED_STAGE_MIN:
                spread = float(q.max() - q.min())
                q_repeats = spread <= 1e-9 + _STUCK_Q_REL_TOL * abs(float(q.median()))

        pos = np.sort(visit_order.loc[idx].to_numpy())
        consecutive = len(pos) >= REPEATED_STAGE_MIN and int(np.max(np.diff(pos))) == 1

        if not (q_repeats or consecutive):
            continue

        n = len(idx)
        if q_repeats:
            reason = (
                f"stage {stage_val:g} and discharge both repeat {n}x "
                f"(likely duplicated / placeholder rows)"
            )
        else:
            reason = (
                f"stage {stage_val:g} repeats {n}x across consecutive visits "
                f"(likely a stuck gauge)"
            )
        mask.loc[idx] = True
        detail.loc[idx] = reason

    return mask, detail


def clean_and_validate_measurements(
    df: pd.DataFrame,
    column_overrides: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Clean and validate a field measurement table for rating curve work.

    ``df`` may use the canonical schema (see :mod:`rating_curve_automater.schema`) or any workbook
    headers that :func:`~rating_curve_automater.schema.resolve_columns` can map. The returned frame
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
    if DISCHARGE_UNCERTAINTY in result.columns:
        result[DISCHARGE_UNCERTAINTY] = clean_uncertainty_series(result[DISCHARGE_UNCERTAINTY])

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

    stuck_mask, stuck_detail = _stuck_stage_mask(result)
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
            reasons.append(str(stuck_detail.get(row.name, "") or "repeated placeholder stage"))
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
    from rating_curve_automater.loader import load_measurements

    canonical, _ = load_measurements(input_path, sheet=sheet_name)
    return canonical


DEFAULT_DATASET = Path(__file__).resolve().parent / "data" / "10_year_single_site_rating_curve_data.xlsx"
DEFAULT_DATASET_SHEET = "Measurements"


def clean_measurements_to_csv(
    input_excel: str | Path,
    output_csv: str | Path,
    sheet_name: str | int | None = None,
    column_overrides: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load a workbook, clean/validate it, and write the result to CSV.

    ``column_overrides`` forces canonical fields to named source columns (e.g.
    ``{"stage_m": "Elev (masl)"}``) when auto-detection picks the wrong one.

    Returns the cleaned dataframe and prints a short summary.
    """
    from rating_curve_automater.loader import load_measurements

    canonical, report = load_measurements(
        input_excel, sheet=sheet_name, column_overrides=column_overrides or None
    )
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
    parser.add_argument("--stage-column", type=str, default=None,
                        help="Force the stage column by header name (overrides auto-detection).")
    parser.add_argument("--discharge-column", type=str, default=None,
                        help="Force the discharge column by header name.")
    parser.add_argument("--date-column", type=str, default=None,
                        help="Force the date column by header name.")
    args = parser.parse_args()

    overrides = {
        canonical: name
        for canonical, name in (
            (STAGE_M, args.stage_column),
            (DISCHARGE_CMS, args.discharge_column),
            (DATE, args.date_column),
        )
        if name
    }

    if args.default_dataset:
        input_excel: str | Path = DEFAULT_DATASET
        sheet_name = args.sheet_name
    elif args.input_excel:
        input_excel = args.input_excel
        sheet_name = args.sheet_name
    else:
        parser.error("provide an input_excel path or --default-dataset")

    clean_measurements_to_csv(input_excel, args.output_csv, sheet_name, column_overrides=overrides)


if __name__ == "__main__":
    main()
