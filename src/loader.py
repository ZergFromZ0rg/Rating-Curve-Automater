"""Flexible measurement loader.

``load_measurements`` accepts .xlsx / .xls / .csv, picks the right sheet and
header row, maps arbitrary column headers onto the canonical schema, converts
stage/discharge to SI, and returns the canonical frame plus a
:class:`LoadReport` describing every choice it made.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.schema import (
    DATE,
    DISCHARGE_CMS,
    REQUIRED_FIELDS,
    STAGE_M,
    ColumnMapping,
    apply_mapping,
    resolve_columns,
)
from src.units import UnitConversion, detect_discharge_unit, detect_stage_unit

MAX_HEADER_SCAN = 15
SAMPLE_ROWS = 5


@dataclass
class LoadReport:
    path: Path
    sheet_name: str | None
    available_sheets: list[str]
    header_row: int
    source_columns: list[str]
    mapping: ColumnMapping
    units: dict[str, UnitConversion]
    n_rows: int
    sample: pd.DataFrame
    sheet_confident: bool = True
    header_confident: bool = True
    two_row_header: bool = False
    messages: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.mapping.is_complete

    @property
    def needs_review(self) -> bool:
        """True when a human should sanity-check the loader's choices."""
        return (
            not self.ok
            or not self.sheet_confident
            or not self.header_confident
            or bool(self.mapping.ambiguous)
        )

    def describe(self) -> str:
        lines = [f"File: {self.path.name}"]
        if self.sheet_name is not None:
            tag = "" if self.sheet_confident else "  [uncertain - check this]"
            lines.append(f"Sheet: {self.sheet_name}  (of {', '.join(self.available_sheets)}){tag}")
        tag = "" if self.header_confident else "  [uncertain - check this]"
        lines.append(f"Header row: {self.header_row + 1}{tag}")
        if self.two_row_header:
            lines.append("  (two-row header combined)")
        lines.append("Column mapping:")
        lines += [f"  - {line}" for line in self.mapping.describe()]
        if self.units:
            lines.append("Units:")
            for canonical, conv in self.units.items():
                kind = "detected" if conv.detected else "assumed"
                lines.append(f"  - {canonical}: {conv.label} ({kind})")
        lines.append(f"Data rows: {self.n_rows}")
        lines += [f"! {m}" for m in self.messages]
        return "\n".join(lines)


def _read_sheet_raw(path: Path, sheet, is_csv: bool) -> pd.DataFrame:
    if is_csv:
        return pd.read_csv(path, header=None, dtype=object)
    return pd.read_excel(path, sheet_name=sheet, header=None, dtype=object)


def _looks_numeric(value: object) -> bool:
    if pd.isna(value):
        return False
    try:
        float(str(value).replace(",", ""))
        return True
    except (TypeError, ValueError):
        return False


def _numeric_ratio(values: list) -> float:
    non_null = [v for v in values if not pd.isna(v)]
    if not non_null:
        return 0.0
    return sum(1 for v in non_null if _looks_numeric(v)) / len(non_null)


def _evaluate_header_row(raw: pd.DataFrame, idx: int) -> dict | None:
    """Score row ``idx`` as a candidate header, or ``None`` if implausible."""
    values = list(raw.iloc[idx])
    cells = ["" if pd.isna(v) else str(v).strip() for v in values]
    if not any(cells):
        return None

    mapping = resolve_columns(cells)
    resolved = sum(1 for f in REQUIRED_FIELDS if mapping.fields.get(f))
    if resolved < 2:
        return None

    positions: list[int] = []
    for canonical in (STAGE_M, DISCHARGE_CMS):
        name = mapping.fields.get(canonical)
        if name and name in cells:
            positions.append(cells.index(name))

    below = raw.iloc[idx + 1: idx + 1 + 12]
    ratios = [
        _numeric_ratio(below.iloc[:, pos].tolist())
        for pos in positions
        if pos < below.shape[1]
    ]
    numeric_score = sum(ratios) / len(ratios) if ratios else 0.0

    return {
        "row": idx,
        "resolved": resolved,
        "mapping": mapping,
        "numeric_score": numeric_score,
        "header_text_ratio": 1.0 - _numeric_ratio(values),
    }


def _detect_header(raw: pd.DataFrame, max_scan: int = MAX_HEADER_SCAN) -> tuple[int, bool, bool]:
    """Return ``(header_row, confident, two_row_header)``."""
    candidates = [
        ev for idx in range(min(max_scan, len(raw)))
        if (ev := _evaluate_header_row(raw, idx)) is not None
    ]
    if not candidates:
        return 0, False, False

    candidates.sort(
        key=lambda e: (
            e["resolved"],
            round(e["numeric_score"], 2),
            round(e["header_text_ratio"], 2),
            -e["row"],
        ),
        reverse=True,
    )
    best = candidates[0]
    confident = best["resolved"] >= 3 and best["numeric_score"] >= 0.5

    two_row = False
    below_1, below_2 = best["row"] + 1, best["row"] + 2
    if below_2 < len(raw):
        if _numeric_ratio(list(raw.iloc[below_1])) < 0.3 and _numeric_ratio(list(raw.iloc[below_2])) >= 0.5:
            two_row = True

    return best["row"], confident, two_row


def _flatten_columns(columns) -> list[str]:
    flat = []
    for col in columns:
        parts = col if isinstance(col, tuple) else (col,)
        kept = [
            str(p).strip()
            for p in parts
            if not pd.isna(p) and not str(p).startswith("Unnamed:")
        ]
        flat.append(" ".join(kept))
    return flat


def _reread_with_header(
    path: Path, sheet, is_csv: bool, header_row: int, two_row: bool = False
) -> pd.DataFrame:
    header: int | list[int] = [header_row, header_row + 1] if two_row else header_row
    if is_csv:
        df = pd.read_csv(path, header=header)
    else:
        df = pd.read_excel(path, sheet_name=sheet, header=header)
    if two_row:
        df.columns = _flatten_columns(df.columns)
    return df


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed:") and str(c).strip()]]
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    return df.reset_index(drop=True)


def _pick_sheet(path: Path, sheets: list[str], max_scan: int) -> tuple[str, bool]:
    scored: list[tuple[str, int, float, int]] = []
    for sheet in sheets:
        raw = _read_sheet_raw(path, sheet, is_csv=False)
        if raw.empty:
            continue
        header_row, _, _ = _detect_header(raw, max_scan)
        ev = _evaluate_header_row(raw, header_row)
        resolved = ev["resolved"] if ev else 0
        numeric = ev["numeric_score"] if ev else 0.0
        data_rows = max(0, len(raw) - header_row - 1)
        scored.append((sheet, resolved, numeric, data_rows))

    if not scored:
        return sheets[0], False

    scored.sort(key=lambda s: (s[1], s[2], s[3]), reverse=True)
    fully_resolved = [s for s in scored if s[1] >= len(REQUIRED_FIELDS)]
    if fully_resolved:
        return fully_resolved[0][0], len(fully_resolved) == 1
    return scored[0][0], False


def load_measurements(
    path: str | Path,
    *,
    sheet: str | int | None = None,
    header_row: int | None = None,
    column_overrides: dict[str, str] | None = None,
    convert_units: bool = True,
) -> tuple[pd.DataFrame, LoadReport]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    is_csv = path.suffix.lower() == ".csv"
    messages: list[str] = []
    sheet_confident = True

    available_sheets: list[str] = []
    chosen_sheet: str | None = None
    if not is_csv:
        try:
            excel = pd.ExcelFile(path)
        except Exception as exc:  # e.g. .xls without xlrd installed
            raise ValueError(f"Could not open {path.name}: {exc}") from exc
        available_sheets = list(excel.sheet_names)
        if sheet is None:
            chosen_sheet, sheet_confident = _pick_sheet(path, available_sheets, MAX_HEADER_SCAN)
            if len(available_sheets) > 1:
                messages.append(f"Auto-selected sheet '{chosen_sheet}'.")
                if not sheet_confident:
                    messages.append("More than one sheet looks like measurement data - verify.")
        elif isinstance(sheet, int):
            chosen_sheet = available_sheets[sheet]
        else:
            if sheet not in available_sheets:
                raise ValueError(f"Sheet '{sheet}' not in workbook ({available_sheets}).")
            chosen_sheet = sheet

    raw = _read_sheet_raw(path, chosen_sheet, is_csv)
    if raw.empty:
        raise ValueError("The selected sheet is empty.")

    if header_row is not None:
        resolved_header, header_confident, two_row = header_row, True, False
    else:
        resolved_header, header_confident, two_row = _detect_header(raw)
        if resolved_header != 0:
            messages.append(f"Auto-detected header on row {resolved_header + 1}.")
        if not header_confident:
            messages.append("Header row is a best guess - verify the column mapping below.")

    df = _clean_frame(
        _reread_with_header(path, chosen_sheet, is_csv, resolved_header, two_row)
    )
    source_columns = list(df.columns)

    mapping = resolve_columns(source_columns, overrides=column_overrides)
    for canonical, cols in mapping.ambiguous.items():
        messages.append(
            f"{canonical}: '{mapping.fields[canonical]}' chosen over {cols[1:]}."
        )

    units: dict[str, UnitConversion] = {}
    if mapping.is_complete:
        canonical_df = apply_mapping(df, mapping)

        stage_conv = detect_stage_unit(mapping.fields[STAGE_M])
        discharge_conv = detect_discharge_unit(mapping.fields[DISCHARGE_CMS])
        units = {STAGE_M: stage_conv, DISCHARGE_CMS: discharge_conv}

        if convert_units:
            for canonical, conv in ((STAGE_M, stage_conv), (DISCHARGE_CMS, discharge_conv)):
                if conv.detected and conv.factor != 1.0:
                    canonical_df[canonical] = pd.to_numeric(
                        canonical_df[canonical], errors="coerce"
                    ) * conv.factor
                    messages.append(f"Converted {canonical} from {conv.label} to SI.")
    else:
        canonical_df = df
        missing = ", ".join(mapping.unresolved_required)
        messages.append(f"Unresolved required column(s): {missing}.")

    report = LoadReport(
        path=path,
        sheet_name=chosen_sheet,
        available_sheets=available_sheets,
        header_row=resolved_header,
        source_columns=source_columns,
        mapping=mapping,
        units=units,
        n_rows=len(canonical_df),
        sample=canonical_df.head(SAMPLE_ROWS).copy(),
        sheet_confident=sheet_confident,
        header_confident=header_confident,
        two_row_header=two_row,
        messages=messages,
    )
    return canonical_df, report
