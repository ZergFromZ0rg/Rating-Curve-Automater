"""Streamlit front end for the Rating Curve Automater.

Run with::

    streamlit run app.py

A thin view over :class:`src.workflow.RatingCurveWorkflow` -- the same headless
controller the Tkinter app (``rating_curve_app.py``) uses.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.loader import load_measurements
from src.rating_curve_plot import make_rating_curve_figure
from src.schema import (
    ALL_FIELDS,
    DATE,
    DISCHARGE_CMS,
    FIELD_LABELS,
    REQUIRED_FIELDS,
    STAGE_M,
)
from src.workflow import DEFAULT_UNCERTAINTY_THRESHOLD, RatingCurveWorkflow

st.set_page_config(page_title="Rating Curve Automater", page_icon="📈", layout="wide")

AUTO = "(auto-detect)"


# --------------------------------------------------------------------------- #
# Cached pipeline steps (data is small; these just avoid recompute on every
# unrelated widget change).
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def probe(file_key: str, path: str, sheet: str | None, header_row: int | None):
    _, report = load_measurements(path, sheet=sheet, header_row=header_row)
    return report


@st.cache_data(show_spinner=False)
def validate(file_key: str, path: str, sheet: str | None, header_row: int | None, overrides: tuple):
    wf = RatingCurveWorkflow()
    result = wf.load_and_validate(
        path, sheet_name=sheet, column_overrides=dict(overrides) or None, header_row=header_row
    )
    return result


@st.cache_data(show_spinner=False)
def fit_and_report(
    file_key: str,
    path: str,
    sheet: str | None,
    header_row: int | None,
    overrides: tuple,
    h0: float | None,
    segments: int,
    site: str | None,
    threshold: float,
):
    wf = RatingCurveWorkflow()
    wf.load_and_validate(
        path, sheet_name=sheet, column_overrides=dict(overrides) or None, header_row=header_row
    )
    outcome = wf.run_fit(h0=h0, segments=segments, site=site)
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as handle:
        out_path = handle.name
    wf.export_report(out_path, uncertainty_threshold=threshold)
    report_bytes = Path(out_path).read_bytes()
    Path(out_path).unlink(missing_ok=True)
    return outcome, wf.fit_df, report_bytes


def _friendly(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={k: v for k, v in FIELD_LABELS.items() if k in df.columns})


# --------------------------------------------------------------------------- #
# 1. Upload
# --------------------------------------------------------------------------- #
st.title("📈 Rating Curve Automater")
st.caption("Upload field gaugings → check the detected layout → fit `Q = a·(H − h0)^b` → download an Excel report.")

uploaded = st.file_uploader("Measurement workbook", type=["xlsx", "xls", "csv"])
if uploaded is None:
    st.info("Upload an `.xlsx`, `.xls` or `.csv` file to begin.")
    st.stop()

data = uploaded.getvalue()
file_key = hashlib.md5(data).hexdigest()
suffix = Path(uploaded.name).suffix or ".xlsx"
if st.session_state.get("file_key") != file_key:
    tmp = Path(tempfile.gettempdir()) / f"rca_{file_key}{suffix}"
    tmp.write_bytes(data)
    st.session_state["file_key"] = file_key
    st.session_state["path"] = str(tmp)
path = st.session_state["path"]


# --------------------------------------------------------------------------- #
# 2. Sheet / header row
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Loading")
    try:
        peek = pd.ExcelFile(path).sheet_names if suffix != ".csv" else []
    except Exception:
        peek = []
    if peek:
        sheet_choice = st.selectbox("Sheet", [AUTO, *peek])
        sheet = None if sheet_choice == AUTO else sheet_choice
    else:
        sheet = None
    header_text = st.text_input("Header row (1 = first row)", value="", placeholder="auto")
    header_row = (int(header_text) - 1) if header_text.strip().isdigit() else None

try:
    base_report = probe(file_key, path, sheet, header_row)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not read the file: {exc}")
    st.stop()


# --------------------------------------------------------------------------- #
# 3. Column mapping
# --------------------------------------------------------------------------- #
st.subheader("Column mapping")
st.caption("Blank fields are auto-detected. Override any that are wrong.")
overrides: dict[str, str] = {}
grid = st.columns(3)
for i, field_name in enumerate(ALL_FIELDS):
    guess = base_report.mapping.fields.get(field_name)
    options = [AUTO, *base_report.source_columns]
    index = options.index(guess) if guess in base_report.source_columns else 0
    label = FIELD_LABELS[field_name] + (" *" if field_name in REQUIRED_FIELDS else "")
    choice = grid[i % 3].selectbox(label, options, index=index, key=f"map_{field_name}")
    if choice != AUTO:
        overrides[field_name] = choice

if len(set(overrides.values())) != len(overrides):
    st.warning("The same column is mapped to more than one field.")

try:
    result = validate(file_key, path, sheet, header_row, tuple(sorted(overrides.items())))
except Exception as exc:  # noqa: BLE001
    st.error(str(exc))
    st.stop()

report = result.load_report


# --------------------------------------------------------------------------- #
# 4. Detected layout + preview
# --------------------------------------------------------------------------- #
with st.expander("Detected layout", expanded=report.needs_review):
    if report.needs_review:
        st.warning("Low confidence somewhere below — sanity-check it.")
    st.text(report.describe())
    st.dataframe(_friendly(report.sample), use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# 5. Validation
# --------------------------------------------------------------------------- #
st.subheader("Validation")
m1, m2, m3 = st.columns(3)
m1.metric("Valid rows", result.valid_count)
m2.metric("Invalid (excluded)", result.invalid_count)
m3.metric("Warnings (kept)", result.warning_count)

cleaned = result.cleaned
if result.invalid_count:
    with st.expander(f"{result.invalid_count} invalid row(s)"):
        cols = [c for c in (DATE, STAGE_M, DISCHARGE_CMS, "validation_notes") if c in cleaned.columns]
        st.dataframe(_friendly(cleaned.loc[~cleaned["is_valid"], cols]), use_container_width=True, hide_index=True)
if result.warning_count:
    with st.expander(f"{result.warning_count} row(s) kept with warnings"):
        cols = [c for c in (DATE, STAGE_M, DISCHARGE_CMS, "warning_notes") if c in cleaned.columns]
        st.dataframe(_friendly(cleaned.loc[cleaned["has_warning"], cols]), use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# 6. Fit
# --------------------------------------------------------------------------- #
st.subheader("Fit")
f1, f2, f3, f4 = st.columns(4)

site = None
if result.is_multi_site:
    site_choice = f1.selectbox("Site", ["(all sites)", *result.sites])
    site = None if site_choice == "(all sites)" else site_choice
elif result.sites:
    f1.caption(f"Site: {result.sites[0]}")

estimate_h0 = f2.checkbox("Estimate h0", value=True)
h0 = None if estimate_h0 else f2.number_input("h0 (m)", value=0.18, step=0.01, format="%.3f")

segments = f3.radio("Segments", [1, 2], format_func=lambda n: "1 — single" if n == 1 else "2 — piecewise")
threshold = f4.slider("Uncertainty threshold", 0.05, 1.0, DEFAULT_UNCERTAINTY_THRESHOLD, 0.05)

try:
    outcome, fit_df, report_bytes = fit_and_report(
        file_key, path, sheet, header_row, tuple(sorted(overrides.items())),
        h0, segments, site, threshold,
    )
except Exception as exc:  # noqa: BLE001
    st.error(f"Fit failed: {exc}")
    st.stop()

p = outcome.params
if not outcome.is_plausible:
    st.error("**This is not a plausible rating curve.**\n\n" + "\n".join(f"- {w}" for w in outcome.warnings))
elif outcome.warnings:
    st.warning("\n".join(f"- {w}" for w in outcome.warnings))
else:
    st.success(f"Fitted: {p['equation']}  |  R² = {p['r_squared']:.4f}")

k1, k2, k3, k4 = st.columns(4)
k1.metric("a", f"{p['a']:.4f}")
k2.metric("b", f"{p['b']:.4f}")
k3.metric("h0 (m)", f"{p['h0']:.3f}", "estimated" if p["h0_estimated"] else "fixed")
k4.metric("R²", f"{p['r_squared']:.4f}")

log_scale = st.checkbox("Log–log axes")
figure = make_rating_curve_figure(
    fit_df, a=p["a"], b=p["b"], h0=p["h0"], log_scale=log_scale, fit=p
)
st.pyplot(figure, use_container_width=True)


# --------------------------------------------------------------------------- #
# 7. Export
# --------------------------------------------------------------------------- #
st.subheader("Export")
name = f"rating_curve_report_{site}.xlsx" if site else "rating_curve_report.xlsx"
st.download_button(
    "⬇︎ Download Excel report",
    data=report_bytes,
    file_name=name,
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
