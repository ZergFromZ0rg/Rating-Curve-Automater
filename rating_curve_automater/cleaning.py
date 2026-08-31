"""Value- and row-level cleaning for messy field spreadsheets.

Handles the things real workbooks contain that a bare ``pd.to_numeric`` /
``pd.to_datetime`` trips over: ``"N/A"`` / ``"--"`` placeholders, thousands
separators, decimal commas, censored values (``"<0.001"``), Excel serial dates,
ambiguous day/month order, separate date + time columns, and total/average
footer rows.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

NA_TOKENS = {
    "", "-", "--", "---", "n/a", "n/a.", "na", "n.a.", "nan", "none", "null",
    ".", "..", "#n/a", "#value!", "#div/0!", "n.d.", "nd", "n/d", "missing",
    "not measured", "no data", "tbd", "tba", "?", "??", "x",
}

_CENSOR_RE = re.compile(r"^\s*([<>~≈≤≥]=?)\s*")
# European grouping needs a comma-decimal tail or 2+ dot groups to be unambiguous.
_GROUPED_EU = re.compile(r"^-?\d{1,3}(\.\d{3})+,\d+$|^-?\d{1,3}(\.\d{3}){2,}$")
_GROUPED_US = re.compile(r"^-?\d{1,3}(,\d{3})+\.\d+$|^-?\d{1,3}(,\d{3}){2,}$")
_DECIMAL_COMMA = re.compile(r"^-?\d+,\d{1,2}$")
_UNIT_TAIL = re.compile(r"[a-zA-Z%°²³/]+\s*$")

_FOOTER_RE = re.compile(
    r"^\s*(total|totals|subtotal|sum|average|avg|mean|median|count|std\.?dev|"
    r"stdev|min|minimum|max|maximum|range|n\s*=|summary|end of (data|record)|"
    r"notes?\s*:)",
    re.IGNORECASE,
)

_EXCEL_SERIAL_MIN, _EXCEL_SERIAL_MAX = 20000, 80000  # ~1954 .. ~2119


def _clean_number_token(text: str) -> tuple[str, bool]:
    """Return (normalised numeric string, was_censored)."""
    token = text.strip()
    censored = bool(_CENSOR_RE.match(token))
    token = _CENSOR_RE.sub("", token)
    token = token.replace(" ", " ").strip()
    token = _UNIT_TAIL.sub("", token).strip()
    token = token.replace(" ", "")

    if _GROUPED_EU.match(token):
        token = token.replace(".", "").replace(",", ".")
    elif _GROUPED_US.match(token):
        token = token.replace(",", "")
    elif _DECIMAL_COMMA.match(token):
        token = token.replace(",", ".")
    elif token.count(",") == 1 and "." not in token:
        # lone comma with 3 trailing digits -> thousands; otherwise decimal
        token = token.replace(",", "" if re.search(r",\d{3}$", token) else ".")

    return token, censored


def clean_numeric_series(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Coerce a column to float, returning (values, censored_mask)."""
    values: list[float] = []
    censored: list[bool] = []
    for raw in series:
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            values.append(np.nan)
            censored.append(False)
            continue
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            values.append(float(raw))
            censored.append(False)
            continue
        text = str(raw).strip()
        if text.lower() in NA_TOKENS:
            values.append(np.nan)
            censored.append(False)
            continue
        token, was_censored = _clean_number_token(text)
        try:
            values.append(float(token))
        except ValueError:
            values.append(np.nan)
            was_censored = False
        censored.append(was_censored)

    index = series.index
    return (
        pd.Series(values, index=index, dtype="float64"),
        pd.Series(censored, index=index, dtype="bool"),
    )


def _looks_like_excel_serial(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() < 0.8:
        return False
    inside = numeric.dropna().between(_EXCEL_SERIAL_MIN, _EXCEL_SERIAL_MAX)
    return bool(len(inside)) and inside.mean() > 0.9


def _to_timedelta(value: object) -> pd.Timedelta:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.Timedelta(0)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return pd.Timedelta(days=float(value))  # fraction-of-day
    td = pd.to_timedelta(str(value), errors="coerce")
    if not pd.isna(td):
        return td
    parsed = pd.to_datetime(str(value), errors="coerce")
    if pd.isna(parsed):
        return pd.Timedelta(0)
    return pd.Timedelta(hours=int(parsed.hour), minutes=int(parsed.minute), seconds=int(parsed.second))


_RANGE_ISO = re.compile(r"^(\d{4}-\d{1,2}-\d{1,2})\s*/\s*\d{1,2}\s*$")
_RANGE_DAYS = re.compile(r"^(\d{1,2})\s*[-/]\s*\d{1,2}\s+([A-Za-z].*)$")
_RANGE_DASH = re.compile(r"\s[–—]\s")


def _preprocess_date_token(value: object) -> object:
    """Reduce a date-range string to its start date.

    ``"2025-10-17/22"`` -> ``"2025-10-17"``; ``"17-22 Dec 2025"`` -> ``"17 Dec 2025"``;
    ``"Dec 15 - Dec 18 2025"`` -> ``"Dec 15"``. Non-strings pass through.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    m = _RANGE_ISO.match(text)
    if m:
        return m.group(1)
    m = _RANGE_DAYS.match(text)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    parts = _RANGE_DASH.split(text)
    if len(parts) == 2 and any(ch.isdigit() for ch in parts[0]):
        return parts[0].strip()
    return text


_DMY_TOKEN = re.compile(r"^\s*(\d{1,2})[/-](\d{1,2})[/-]\d{2,4}\s*$")


def _has_dayfirst_evidence(series: pd.Series) -> bool:
    """True when a slash/dash date column has a first field > 12 somewhere
    (only valid as a day) and no clear month-first counter-evidence."""
    dayfirst = monthfirst = 0
    for value in series:
        if not isinstance(value, str):
            continue
        m = _DMY_TOKEN.match(value)
        if not m:
            continue
        first, second = int(m.group(1)), int(m.group(2))
        if first > 12 >= second:
            dayfirst += 1
        elif second > 12 >= first:
            monthfirst += 1
    return dayfirst > 0 and dayfirst >= monthfirst


def _parse_mixed(series: pd.Series, *, dayfirst: bool) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=dayfirst)
        except (ValueError, TypeError):
            return pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)


def coerce_datetime(dates: pd.Series, times: pd.Series | None = None) -> pd.Series:
    """Parse a date column (Excel serials, ``DD-Mon-YY``, ambiguous day/month,
    date ranges), optionally adding a separate time column."""
    dates = dates if isinstance(dates, pd.Series) else pd.Series(dates)
    if _looks_like_excel_serial(dates):
        parsed = pd.to_datetime(
            pd.to_numeric(dates, errors="coerce"),
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )
    else:
        pre = dates.map(_preprocess_date_token)
        plain = _parse_mixed(pre, dayfirst=False)
        dayfirst = _parse_mixed(pre, dayfirst=True)
        if dayfirst.notna().sum() > plain.notna().sum() or _has_dayfirst_evidence(pre):
            parsed = dayfirst
        else:
            parsed = plain

    if times is not None:
        offsets = pd.to_timedelta(
            pd.Series([_to_timedelta(v) for v in times], index=parsed.index)
        )
        parsed = parsed + offsets  # NaT + timedelta stays NaT

    return parsed


def is_footer_label(value: object) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return bool(_FOOTER_RE.match(str(value)))


def drop_footer_rows(df: pd.DataFrame, label_columns: list[str]) -> tuple[pd.DataFrame, int]:
    """Drop trailing total/average/summary rows detected in ``label_columns``."""
    if df.empty or not label_columns:
        return df, 0
    mask = pd.Series(False, index=df.index)
    for col in label_columns:
        if col in df.columns:
            mask |= df[col].map(is_footer_label)
    if not mask.any():
        return df, 0
    return df.loc[~mask].reset_index(drop=True), int(mask.sum())
