"""Detect and unpivot wide multi-station measurement layouts.

Some workbooks put two or more stations side by side::

    Date | SW1 readings | Stage above bed, m | Q, cms | SW2 readings | Stage above bed, m | Q, cms

:func:`reshape_wide_stations` turns that into long form with a ``site`` column.
"""

from __future__ import annotations

import re

import pandas as pd

from rating_curve_automater.schema import (
    DATE,
    DISCHARGE_CMS,
    SITE,
    STAGE_M,
    field_scores_for,
    resolve_columns,
)

_MATCH_THRESHOLD = 55
_GENERIC_LABEL_WORDS = re.compile(
    r"(?i)\b(readings?|reading|gauge|gage|stage|level|data|measurements?|obs|"
    r"observations?|values?|record|column|staff|meter|m)\b"
)


def _station_name(label: object, index: int) -> str:
    if label is not None and not (isinstance(label, float) and pd.isna(label)):
        name = _GENERIC_LABEL_WORDS.sub("", str(label))
        name = re.sub(r"[\s,;:_.\-]+", " ", name).strip()
        if name:
            return name
    return f"Station {index + 1}"


def detect_wide_blocks(columns: list) -> list[tuple[str, object, object, object]] | None:
    """Return ``[(station_name, stage_col, discharge_col, label_col), ...]`` or ``None``."""
    scores = {c: field_scores_for(c) for c in columns}

    stage_cols = [
        c for c in columns
        if scores[c][STAGE_M] >= _MATCH_THRESHOLD and scores[c][STAGE_M] >= scores[c][DISCHARGE_CMS]
    ]
    disch_cols = [
        c for c in columns
        if scores[c][DISCHARGE_CMS] >= _MATCH_THRESHOLD and scores[c][DISCHARGE_CMS] > scores[c][STAGE_M]
    ]
    if len(stage_cols) < 2 or len(stage_cols) != len(disch_cols):
        return None

    pos = {c: i for i, c in enumerate(columns)}
    stage_cols.sort(key=pos.get)
    disch_cols.sort(key=pos.get)

    for k, stage_col in enumerate(stage_cols):
        disch_col = disch_cols[k]
        next_stage = pos[stage_cols[k + 1]] if k + 1 < len(stage_cols) else len(columns)
        if not pos[stage_col] < pos[disch_col] < next_stage:
            return None

    blocks: list[tuple[str, object, object, object]] = []
    for k, (stage_col, disch_col) in enumerate(zip(stage_cols, disch_cols)):
        start = -1 if k == 0 else pos[stage_cols[k - 1]]
        between = [
            c for c in columns
            if start < pos[c] < pos[stage_col] and max(scores[c].values()) < _MATCH_THRESHOLD
        ]
        label = between[-1] if between else None
        blocks.append((_station_name(label, k), stage_col, disch_col, label))
    return blocks


def reshape_wide_stations(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, object]] | None:
    """Unpivot a wide multi-station frame.

    Returns ``(long_df, station_names, unit_reference_columns)`` or ``None`` when
    the layout is not wide. ``unit_reference_columns`` maps ``stage_m`` /
    ``discharge_cms`` to an original header so units can still be detected.
    """
    blocks = detect_wide_blocks(list(df.columns))
    if blocks is None:
        return None

    date_col = resolve_columns(df.columns).fields.get(DATE)
    consumed = {c for _, sc, dc, lbl in blocks for c in (sc, dc, lbl) if c is not None}
    shared = [c for c in df.columns if c not in consumed and c != date_col]

    frames = []
    for name, stage_col, disch_col, _ in blocks:
        cols = ([date_col] if date_col else []) + [stage_col, disch_col] + shared
        sub = df[cols].copy()
        rename = {stage_col: STAGE_M, disch_col: DISCHARGE_CMS}
        if date_col:
            rename[date_col] = DATE
        sub = sub.rename(columns=rename)
        sub[SITE] = name
        frames.append(sub)

    long_df = pd.concat(frames, ignore_index=True)
    unit_ref = {STAGE_M: blocks[0][1], DISCHARGE_CMS: blocks[0][2]}
    return long_df, [b[0] for b in blocks], unit_ref
