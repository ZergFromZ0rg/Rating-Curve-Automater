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
    messages: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.mapping.is_complete

    def describe(self) -> str:
        lines = [f"File: {self.path.name}"]
        if self.sheet_name is not None:
            lines.append(f"Sheet: {self.sheet_name}  (of {', '.join(self.available_sheets)})")
        lines.append(f"Header row: {self.header_row + 1}")
        lines.append("Column mapping:")
        lines += [f"  - {line}" for line in self.mapping.describe()]
        if self.units:
            lines.append("Units:")
            for canonical, conv in self.units.items():
                tag = "detected" if conv.detected else "assumed"
                lines.append(f"  - {canonical}: {conv.label} ({tag})")
        lines.append(f"Data rows: {self.n_rows}")
        lines += [f"! {m}" for m in self.messages]
        return "\n".join(lines)


def _read_sheet_raw(path: Path, sheet, is_csv: bool) -> pd.DataFrame:
    if is_csv:
        return pd.read_csv(path, header=None, dtype=object)
    return pd.read_excel(path, sheet_name=sheet, header=None, dtype=object)


def _score_header_row(raw: pd.DataFrame, row_idx: int) -> tuple[int, int]:
    """(#required fields resolvable, #non-null text cells) for row as header."""
    header = list(raw.iloc[row_idx])
    if all(pd.isna(v) for v in header):
        return (0, 0)
    mapping = resolve_columns([("" if pd.isna(v) else str(v)) for v in header])
    resolved = sum(1 for f in REQUIRED_FIELDS if mapping.fields.get(f))
    text_cells = sum(1 for v in header if isinstance(v, str) and v.strip() and not _looks_numeric(v))
    return (resolved, text_cells)


def _looks_numeric(value: object) -> bool:
    try:
        float(str(value))
        return True
    except (TypeError, ValueError):
        return False


def _detect_header_row(raw: pd.DataFrame, max_scan: int = MAX_HEADER_SCAN) -> int:
    best_row, best_key = 0, (-1, -1)
    for row_idx in range(min(max_scan, len(raw))):
        resolved, text_cells = _score_header_row(raw, row_idx)
        key = (resolved, text_cells)
        if resolved >= 2 and key > best_key:
            best_row, best_key = row_idx, key
    return best_row


def _reread_with_header(path: Path, sheet, is_csv: bool, header_row: int) -> pd.DataFrame:
    if is_csv:
        return pd.read_csv(path, header=header_row)
    return pd.read_excel(path, sheet_name=sheet, header=header_row)


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed:")]]
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    return df.reset_index(drop=True)


def _pick_sheet(path: Path, sheets: list[str], max_scan: int) -> str:
    best_sheet, best_key = sheets[0], (-1, -1)
    for sheet in sheets:
        raw = _read_sheet_raw(path, sheet, is_csv=False)
        if raw.empty:
            continue
        header_row = _detect_header_row(raw, max_scan)
        resolved, text_cells = _score_header_row(raw, header_row)
        key = (resolved, len(raw))
        if key > best_key:
            best_sheet, best_key = sheet, key
    return best_sheet


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

    available_sheets: list[str] = []
    chosen_sheet: str | None = None
    if not is_csv:
        try:
            excel = pd.ExcelFile(path)
        except Exception as exc:  # e.g. .xls without xlrd installed
            raise ValueError(f"Could not open {path.name}: {exc}") from exc
        available_sheets = list(excel.sheet_names)
        if sheet is None:
            chosen_sheet = _pick_sheet(path, available_sheets, MAX_HEADER_SCAN)
            if len(available_sheets) > 1:
                messages.append(f"Auto-selected sheet '{chosen_sheet}'.")
        elif isinstance(sheet, int):
            chosen_sheet = available_sheets[sheet]
        else:
            if sheet not in available_sheets:
                raise ValueError(f"Sheet '{sheet}' not in workbook ({available_sheets}).")
            chosen_sheet = sheet

    raw = _read_sheet_raw(path, chosen_sheet, is_csv)
    if raw.empty:
        raise ValueError("The selected sheet is empty.")

    resolved_header = header_row if header_row is not None else _detect_header_row(raw)
    if header_row is None and resolved_header != 0:
        messages.append(f"Auto-detected header on row {resolved_header + 1}.")

    df = _clean_frame(_reread_with_header(path, chosen_sheet, is_csv, resolved_header))
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
        messages=messages,
    )
    return canonical_df, report
