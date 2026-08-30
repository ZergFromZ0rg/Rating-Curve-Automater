"""Canonical measurement schema and column resolution.

Every module downstream of loading works with these canonical column names, so
input workbooks can use whatever headers they like. :func:`resolve_columns`
maps a workbook's real headers onto the canonical names (with optional manual
overrides), and :func:`apply_mapping` renames a frame accordingly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

# --- Canonical column names -------------------------------------------------

DATE = "date"
STAGE_M = "stage_m"
DISCHARGE_CMS = "discharge_cms"
QUALITY = "quality"
NOTES = "notes"
SITE = "site"

REQUIRED_FIELDS: tuple[str, ...] = (DATE, STAGE_M, DISCHARGE_CMS)
OPTIONAL_FIELDS: tuple[str, ...] = (QUALITY, NOTES, SITE)
ALL_FIELDS: tuple[str, ...] = REQUIRED_FIELDS + OPTIONAL_FIELDS

#: Friendly labels for messages, plots and report headings.
FIELD_LABELS: dict[str, str] = {
    DATE: "Date",
    STAGE_M: "Stage (m)",
    DISCHARGE_CMS: "Discharge (m³/s)",
    QUALITY: "Quality",
    NOTES: "Field notes",
    SITE: "Site",
}

#: Header aliases per canonical field (matched against a normalised header).
FIELD_ALIASES: dict[str, list[str]] = {
    DATE: [
        "date", "datetime", "date time", "timestamp", "time stamp",
        "measurement date", "measured date", "obs date", "observation date",
        "sample date", "survey date", "gauging date", "day",
    ],
    STAGE_M: [
        "stage", "stage above bed", "stage height", "stage m", "river stage",
        "gauge height", "gage height", "gauge reading", "gh", "sg",
        "water level", "water surface level", "water surface elevation", "wse",
        "water elevation", "level", "elevation", "elev", "head", "h", "hg",
        "staff gauge", "stilling well",
    ],
    DISCHARGE_CMS: [
        "discharge", "measured discharge", "measured discharge q", "q meas",
        "qmeas", "gauged discharge", "q", "flow", "streamflow", "stream flow",
        "flow rate", "river flow", "total discharge", "qm",
    ],
    QUALITY: [
        "quality", "field quality", "data quality", "measurement quality",
        "qual", "grade", "rating", "quality code", "quality flag", "qa qc",
    ],
    NOTES: [
        "field notes", "notes", "note", "comment", "comments", "remarks",
        "remark", "observations", "description", "field comment",
    ],
    SITE: [
        "site", "station", "site name", "station name", "site id", "station id",
        "site no", "station no", "gauge", "gauging station", "location", "river",
    ],
}

_SHORT_ALIASES = {a for aliases in FIELD_ALIASES.values() for a in aliases if len(a) <= 2}

_BRACKET_RE = re.compile(r"[\(\[\{].*?[\)\]\}]")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_header(name: object) -> str:
    """Lower-case, drop unit brackets and punctuation, collapse whitespace."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    text = str(name).strip().lower().replace("³", "3").replace("²", "2")
    text = _BRACKET_RE.sub(" ", text)
    text = _NON_ALNUM_RE.sub(" ", text)
    return " ".join(text.split())


def _alias_score(header_norm: str, alias: str) -> int:
    if not header_norm or not alias:
        return 0
    if header_norm == alias:
        return 100

    header_tokens = header_norm.split()
    alias_tokens = alias.split()

    if len(alias) <= 2:
        # Single/double-letter aliases ("q", "h", "gh") are noisy: only accept
        # them when the header is essentially just that token.
        return 92 if alias in header_tokens and len(header_tokens) <= 2 else 0

    if set(alias_tokens).issubset(header_tokens):
        return 80 if alias in header_norm else 70

    if alias in header_norm:
        return 55

    overlap = len(set(alias_tokens) & set(header_tokens)) / len(alias_tokens)
    return int(45 * overlap) if overlap >= 0.75 else 0


def _field_candidates(
    field_name: str,
    columns: Iterable[str],
    normalized: dict[str, str],
    taken: set[str],
    threshold: int = 50,
) -> list[tuple[str, int]]:
    aliases = FIELD_ALIASES[field_name]
    scored: list[tuple[str, int]] = []
    for col in columns:
        if col in taken:
            continue
        score = max((_alias_score(normalized[col], alias) for alias in aliases), default=0)
        if score >= threshold:
            scored.append((col, score))
    scored.sort(key=lambda item: (-item[1], list(columns).index(item[0])))
    return scored


@dataclass
class ColumnMapping:
    """Result of matching a workbook's headers to the canonical schema."""

    fields: dict[str, str | None]
    scores: dict[str, int] = field(default_factory=dict)
    ambiguous: dict[str, list[str]] = field(default_factory=dict)

    @property
    def unresolved_required(self) -> list[str]:
        return [f for f in REQUIRED_FIELDS if not self.fields.get(f)]

    @property
    def is_complete(self) -> bool:
        return not self.unresolved_required

    def source_for(self, canonical: str) -> str | None:
        return self.fields.get(canonical)

    def describe(self) -> list[str]:
        lines = []
        for canonical in ALL_FIELDS:
            src = self.fields.get(canonical)
            if src is None:
                if canonical in REQUIRED_FIELDS:
                    lines.append(f"{FIELD_LABELS[canonical]}: NOT FOUND")
                continue
            note = ""
            if canonical in self.ambiguous:
                others = [c for c in self.ambiguous[canonical] if c != src]
                if others:
                    note = f"  (also matched: {', '.join(map(str, others))})"
            lines.append(f"{FIELD_LABELS[canonical]}: \"{src}\"{note}")
        return lines


def resolve_columns(
    columns: Iterable[str],
    *,
    overrides: dict[str, str] | None = None,
) -> ColumnMapping:
    """Map workbook headers onto canonical field names.

    ``overrides`` forces a canonical field to a specific source column and
    bypasses matching for that field.
    """
    columns = list(columns)
    overrides = overrides or {}
    normalized = {col: normalize_header(col) for col in columns}

    fields: dict[str, str | None] = {}
    scores: dict[str, int] = {}
    ambiguous: dict[str, list[str]] = {}
    taken: set[str] = set()

    # Honour overrides first so they never get claimed by another field.
    for canonical, src in overrides.items():
        if canonical in ALL_FIELDS and src in columns:
            fields[canonical] = src
            scores[canonical] = 100
            taken.add(src)

    for canonical in ALL_FIELDS:
        if canonical in fields:
            continue
        candidates = _field_candidates(canonical, columns, normalized, taken)
        if not candidates:
            fields[canonical] = None
            continue
        best_col, best_score = candidates[0]
        fields[canonical] = best_col
        scores[canonical] = best_score
        taken.add(best_col)
        close = [c for c, s in candidates[1:] if s >= 70 and best_score - s <= 15]
        if close:
            ambiguous[canonical] = [best_col, *close]

    return ColumnMapping(fields=fields, scores=scores, ambiguous=ambiguous)


def has_canonical_columns(df: pd.DataFrame) -> bool:
    return all(name in df.columns for name in REQUIRED_FIELDS)


def apply_mapping(df: pd.DataFrame, mapping: ColumnMapping) -> pd.DataFrame:
    """Return ``df`` with mapped columns renamed to canonical names.

    Canonical columns are ordered first; unmapped columns are kept as-is.
    """
    rename = {src: canonical for canonical, src in mapping.fields.items() if src is not None}
    out = df.rename(columns=rename)
    canonical_cols = [c for c in ALL_FIELDS if c in out.columns]
    other_cols = [c for c in out.columns if c not in canonical_cols]
    return out[canonical_cols + other_cols]


def ensure_canonical(
    df: pd.DataFrame,
    *,
    overrides: dict[str, str] | None = None,
    required: Iterable[str] = REQUIRED_FIELDS,
) -> pd.DataFrame:
    """Canonicalise ``df`` unless the ``required`` columns are already present.

    Idempotent; used defensively by functions that may be handed either a raw
    or an already-processed frame. ``required`` lets callers that only need a
    subset (e.g. the fitter needs stage + discharge, not date) pass a narrower
    set.
    """
    required = tuple(required)
    if all(name in df.columns for name in required):
        return df
    mapping = resolve_columns(df.columns, overrides=overrides)
    unresolved = [f for f in required if not mapping.fields.get(f)]
    if unresolved:
        missing = ", ".join(FIELD_LABELS[f] for f in unresolved)
        raise ValueError(
            f"Could not identify required column(s): {missing}. "
            f"Found headers: {list(df.columns)}"
        )
    return apply_mapping(df, mapping)
