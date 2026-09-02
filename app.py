"""Streamlit front end for the Rating Curve Automater.

Run with::

    streamlit run app.py

A thin view over :class:`rating_curve_automater.workflow.RatingCurveWorkflow`, the
headless load -> validate -> fit -> export controller.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from rating_curve_automater.loader import load_measurements
from rating_curve_automater.rating_curve_plot import make_rating_curve_figure, make_residual_time_figure
from rating_curve_automater.schema import (
    ALL_FIELDS,
    DATE,
    DISCHARGE_CMS,
    FIELD_LABELS,
    REQUIRED_FIELDS,
    STAGE_M,
)
from rating_curve_automater.rating_curve_fitting import DEFAULT_DISCHARGE_UNCERTAINTY_PCT
from rating_curve_automater.rating_table import DEFAULT_STAGE_STEP_M
from rating_curve_automater.workflow import DEFAULT_UNCERTAINTY_THRESHOLD, RatingCurveWorkflow

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
    uncertainty_pct: float,
    rating_step: float,
    method: str,
    bayesian_sampler: str = "auto",
    section_csv: str | None = None,
    section_slope: float = 0.0,
    section_n: float | None = None,
    section_offset: float = 0.0,
):
    wf = RatingCurveWorkflow()
    wf.load_and_validate(
        path, sheet_name=sheet, column_overrides=dict(overrides) or None, header_row=header_row
    )
    outcome = wf.run_fit(
        h0=h0, segments=segments, site=site,
        discharge_uncertainty_pct=uncertainty_pct, method=method,
        bayesian_sampler=bayesian_sampler,
    )
    if section_csv and section_slope > 0:
        try:
            wf.manning_check(section_csv, section_slope, mannings_n=section_n,
                             stage_offset=section_offset)
        except Exception as exc:  # noqa: BLE001
            outcome.params.setdefault("manning", {"flag": "unusable", "message": str(exc)})
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as handle:
        out_path = handle.name
    wf.export_report(out_path, uncertainty_threshold=threshold, rating_table_step=rating_step)
    report_bytes = Path(out_path).read_bytes()
    Path(out_path).unlink(missing_ok=True)
    rating_table = wf.rating_table(step=rating_step)
    rating_csv = rating_table.to_csv(index=False).encode("utf-8")
    return outcome, wf.fit_df, report_bytes, rating_table, rating_csv


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

method_label = st.radio(
    "Fit method",
    ["Least squares", "Bayesian"],
    horizontal=True,
    help=(
        "Least squares: fast log-log regression (weighted when a discharge-uncertainty "
        "column varies). Bayesian: thodson-usgs `ratingcurve` (PyMC) — needs "
        "`pip install \"rating-curve-automater[bayesian]\"`; the first fit takes ~1 min."
    ),
)
method = "bayesian" if method_label == "Bayesian" else "ols"

bayesian_sampler = "auto"
if method == "bayesian":
    bayesian_sampler = st.radio(
        "Sampler",
        ["auto", "nuts", "advi"],
        horizontal=True,
        format_func=lambda s: {"auto": "auto (NUTS ≤200 gaugings)", "nuts": "NUTS (exact, slow)",
                               "advi": "ADVI (variational, fast)"}[s],
        help="NUTS samples the exact posterior and places breakpoints more reliably; "
             "ADVI is a fast approximation. 'auto' uses NUTS for small records.",
    )

f1, f2, f3, f4, f5 = st.columns(5)

site = None
if result.is_multi_site:
    site_choice = f1.selectbox("Site", ["(all sites)", *result.sites])
    site = None if site_choice == "(all sites)" else site_choice
elif result.sites:
    f1.caption(f"Site: {result.sites[0]}")

estimate_h0 = f2.checkbox("Estimate h0", value=True)
h0 = None if estimate_h0 else f2.number_input("h0 (m)", value=0.18, step=0.01, format="%.3f")

segments = f3.radio(
    "Segments",
    [1, 2, 3, "auto"],
    format_func=lambda n: {1: "1 — single", 2: "2 — piecewise", 3: "3 — piecewise", "auto": "auto (BIC)"}[n],
    help="Number of joined power-law segments, or 'auto' to let BIC pick 1–4.",
)
threshold = f4.slider("Uncertainty threshold", 0.05, 1.0, DEFAULT_UNCERTAINTY_THRESHOLD, 0.05)
uncertainty_pct = f5.number_input(
    "Discharge uncertainty (±%)",
    min_value=0.5, max_value=100.0, value=float(DEFAULT_DISCHARGE_UNCERTAINTY_PCT), step=0.5,
    help=(
        "Assumed measurement uncertainty for gaugings with no value in a "
        "'Discharge uncertainty (±%)' column. Map that column above to weight the "
        "fit point-by-point — noisier gaugings then pull the curve less."
    ),
)
rating_step = f5.number_input(
    "Rating table step (m)",
    min_value=0.001, max_value=1.0, value=float(DEFAULT_STAGE_STEP_M), step=0.005, format="%.3f",
    help="Stage increment for the stage→discharge lookup table (Excel sheet + CSV download).",
)

section_csv = section_slope = section_n = section_offset = None
with st.expander("Manning cross-section check (optional — sanity-checks the extrapolation)"):
    st.caption(
        "Upload a surveyed cross-section (offset + elevation columns) and give the "
        "channel slope. The tool computes an independent Manning curve and flags "
        "where the fitted power law's extrapolation above the highest gauging "
        "disagrees with the channel geometry."
    )
    sec_file = st.file_uploader("Cross-section CSV", type=["csv"], key="xsec")
    mc1, mc2, mc3 = st.columns(3)
    section_slope = mc1.number_input("Channel slope (m/m)", min_value=0.0, value=0.0,
                                     step=0.0001, format="%.5f")
    section_n = mc2.number_input("Manning's n (0 = calibrate to rating)", min_value=0.0,
                                 max_value=0.3, value=0.0, step=0.005, format="%.3f")
    section_offset = mc3.number_input("Stage → WSE offset (m)", value=0.0, step=0.01, format="%.3f",
                                      help="Water-surface elevation = stage H + this offset, in the section's datum.")
    if sec_file is not None and section_slope > 0:
        sec_path = Path(tempfile.gettempdir()) / f"rca_xsec_{hashlib.md5(sec_file.getvalue()).hexdigest()}.csv"
        sec_path.write_bytes(sec_file.getvalue())
        section_csv = str(sec_path)
    elif sec_file is not None:
        st.warning("Enter a positive channel slope to run the check.")

try:
    with st.spinner("Sampling the posterior… (~1 min on the first Bayesian fit)" if method == "bayesian"
                    else "Fitting…"):
        outcome, fit_df, report_bytes, rating_table, rating_csv = fit_and_report(
            file_key, path, sheet, header_row, tuple(sorted(overrides.items())),
            h0, segments, site, threshold, uncertainty_pct, rating_step, method,
            bayesian_sampler,
            section_csv, float(section_slope or 0.0),
            (section_n or None), float(section_offset or 0.0),
        )
except ImportError as exc:
    st.error(str(exc))
    st.stop()
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

bands = p.get("bands")
k1, k2, k3, k4 = st.columns(4)
b_delta = None
if bands and bands.get("b_ci"):
    b_delta = f"{int(round(bands['level'] * 100))}% CI [{bands['b_ci'][0]:.3f}, {bands['b_ci'][1]:.3f}]"
k1.metric("a", f"{p['a']:.4f}")
k2.metric("b", f"{p['b']:.4f}", b_delta, delta_color="off")
h0_note = "fixed"
if p["h0_estimated"]:
    hd = p.get("h0_diagnostics") or {}
    h0_note = "weakly identified" if hd.get("railed") else f"estimated ({hd.get('method', '?')})"
k3.metric("h0 (m)", f"{p['h0']:.3f}", h0_note, delta_color="off")
r2_label = "weighted R²" if p.get("weighted") else "R²"
r2_value = p.get("r_squared_weighted") if p.get("weighted") else p["r_squared"]
k4.metric(r2_label, f"{r2_value:.4f}")

if p.get("method") == "bayesian":
    bx = p.get("bayes", {})
    note = bx.get("auto_segments_note")
    st.caption(f"🔬 Bayesian fit (thodson-usgs `ratingcurve`, PyMC {bx.get('sampler', '?').upper()})."
               + (f" {note}" if note else ""))

if bands:
    pct = int(round(bands["level"] * 100))
    src = f"{bands['n_success']} posterior draws" if bands.get("kind") == "posterior" else f"{bands['n_success']} bootstrap refits"
    st.caption(
        f"Shaded bands: {pct}% confidence (how well the mean curve is pinned down) and "
        f"{pct}% prediction (where a new gauging would fall), from {src}. "
        f"Confidence band is ±{bands['ci_halfwidth_pct_at_median']:.1f}% "
        f"at the median stage. Bands cover the observed stage range only."
        + (f" h0 re-estimated per replicate — 95% CI [{bands['h0_ci'][0]:.3f}, {bands['h0_ci'][1]:.3f}] m."
           if bands.get("h0_ci") else "")
    )
else:
    st.caption("Confidence/prediction bands need at least 4 usable gaugings.")

if p.get("weighted"):
    st.caption(
        f"⚖︎ Weighted least-squares fit using the per-point discharge uncertainty "
        f"column (mean ±{p['mean_uncertainty_pct']:.1f}%)."
    )
elif p.get("uncertainty_source") == "column":
    st.caption("Per-point discharge uncertainty column found, but all values are equal — fit not re-weighted.")
else:
    st.caption(f"Discharge uncertainty assumed at ±{p['uncertainty_pct_default']:.1f}% for every gauging (fit not re-weighted).")

log_scale = st.checkbox("Log–log axes")
figure = make_rating_curve_figure(
    fit_df, a=p["a"], b=p["b"], h0=p["h0"], log_scale=log_scale, fit=p
)
st.pyplot(figure, use_container_width=True)

drift = p.get("drift")
if drift:
    if drift["flag"] == "likely":
        st.warning(f"⏳ {drift['message']}")
    elif drift["flag"] in ("possible", "unassessable"):
        st.info(f"⏳ {drift['message']}")
    else:
        st.caption(f"⏳ {drift['message']} (spans {drift['date_min']} → {drift['date_max']})")
    cp = drift.get("changepoint")
    if cp is not None:
        st.caption(
            f"Estimated changepoint: **{cp['date']}** "
            f"({cp['shift_pct']:+.0f}% across the break, p={cp['p_value']:.3f}; "
            f"{cp['n_before']} gaugings before, {cp['n_after']} after)"
        )
    resid_fig = make_residual_time_figure(fit_df, p)
    if resid_fig is not None:
        with st.expander("Residuals over time", expanded=drift["flag"] != "none"):
            st.pyplot(resid_fig, use_container_width=True)

mc = p.get("manning")
if mc:
    if mc.get("flag") in ("diverges", "implausible-n"):
        st.warning(f"📐 {mc['message']}")
    elif mc.get("flag") in ("check", "unusable"):
        st.info(f"📐 {mc['message']}")
    else:
        st.caption(f"📐 {mc['message']}")


# --------------------------------------------------------------------------- #
# 7. Export
# --------------------------------------------------------------------------- #
st.subheader("Export")
suffix = f"_{site}" if site else ""
e1, e2 = st.columns(2)
e1.download_button(
    "⬇︎ Download Excel report",
    data=report_bytes,
    file_name=f"rating_curve_report{suffix}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
e2.download_button(
    "⬇︎ Download rating table (CSV)",
    data=rating_csv,
    file_name=f"rating_table{suffix}.csv",
    mime="text/csv",
)

with st.expander(f"Rating table preview — stage → discharge every {rating_step:g} m ({len(rating_table)} rows)"):
    st.dataframe(rating_table, use_container_width=True, hide_index=True)
