"""Streamlit front end for the Rating Curve Automater.

Launch it with ``rca app`` (from any install), which runs ``streamlit run`` on
this file.

The interface keeps the load -> validate -> fit -> export flow visible without
making the user navigate a sidebar. It is a thin view over
:class:`rating_curve_automater.workflow.RatingCurveWorkflow`.
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

st.markdown(
    """
    <style>
    :root {
        --rca-bg: #0d141d;
        --rca-panel: #172230;
        --rca-panel-2: #1d2a39;
        --rca-border: #2d4054;
        --rca-muted: #9eb0c4;
        --rca-blue: #61a8ff;
        --rca-green: #45d39a;
    }
    .stApp { background: var(--rca-bg); color: #eef5ff; }
    .block-container { max-width: 1280px; padding-top: 1.65rem; padding-bottom: 1rem; }
    [data-testid="stVerticalBlock"] { gap: .52rem; }
    [data-testid="stHorizontalBlock"] { gap: .8rem; }
    h1, h2, h3 { letter-spacing: -0.03em; }
    h1 { font-size: 1.9rem !important; margin-bottom: 0.05rem !important; }
    h2 { font-size: 1.45rem !important; }
    h3 { font-size: 1.15rem !important; }
    p, label, .stCaption, [data-testid="stMarkdownContainer"] { color: #c6d2e2; }
    .rca-hero { margin-bottom:.5rem; }
    .rca-hero-copy { max-width: 820px; }
    .rca-kicker { color: var(--rca-blue); font-size:0.82rem; font-weight:700; letter-spacing:0.13em; text-transform:uppercase; margin-bottom:0.45rem; }
    .rca-subtitle { color:#aebed0; font-size:1.08rem; line-height:1.45; margin:0; }
    .rca-section-label { color:#f3f7fd; font-size:1.35rem; font-weight:700; margin-bottom:.15rem; }
    .rca-section-help { color:var(--rca-muted); margin-bottom:.55rem; }
    .rca-check { color:#b9c8d8; margin:.48rem 0; }
    .rca-check::first-letter { color:var(--rca-green); }
    [data-testid="stVerticalBlockBorderWrapper"] { background:rgba(23,34,48,.88); border-color:var(--rca-border); border-radius:10px; }
    [data-testid="stVerticalBlockBorderWrapper"] > div { padding-top:.65rem; padding-bottom:.65rem; }
    [data-testid="stFileUploader"] { background:rgba(29,42,57,.9); border:1px dashed #7389a1; border-radius:10px; padding:.7rem; }
    [data-testid="stFileUploaderDropzone"] { background:transparent; }
    [data-testid="stMetric"] { background:rgba(29,42,57,.9); border:1px solid var(--rca-border); border-radius:7px; padding:.45rem .65rem; }
    [data-testid="stMetricLabel"] { color:#aebed0; }
    [data-testid="stMetricValue"] { color:#f1f6fd; }
    .stButton > button, .stDownloadButton > button { width:auto !important; min-width:0 !important; border-radius:6px; border:1px solid #3977c4; background:#2478e5; color:white; font-weight:650; min-height:2.25rem; padding:.35rem .8rem; }
    .stButton > button:hover, .stDownloadButton > button:hover { border-color:#7eb9ff; background:#3188f4; color:white; }
    [data-testid="stExpander"] { border-color:var(--rca-border); background:rgba(18,28,40,.65); }
    hr { border-color:var(--rca-border); }
    @media (max-width: 760px) {
        .block-container { padding-left: .8rem; padding-right: .8rem; }
        h1 { font-size: 1.65rem !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
    return wf.load_and_validate(
        path, sheet_name=sheet, column_overrides=dict(overrides) or None, header_row=header_row
    )


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
    fixed_b: float | None = None,
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
        bayesian_sampler=bayesian_sampler, fixed_b=fixed_b,
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
# 1 · Upload
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div class="rca-hero">
      <div class="rca-hero-copy">
        <h1>Rating Curve Automater</h1>
        <p class="rca-subtitle">Upload measurements, fit a rating curve, and export the results.</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown('<div class="rca-section-label">Upload field measurements</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="rca-section-help">Provide a spreadsheet of stage and discharge measurements. The data is validated before fitting.</div>',
        unsafe_allow_html=True,
    )
    upload_col, status_col = st.columns([1.05, 0.95], gap="large")
    with upload_col:
        uploaded = st.file_uploader(
            "Drag and drop your file here or click to browse",
            type=["xlsx", "xls", "csv"],
            label_visibility="visible",
            help="Messy headers, extra sheets, unit labels, placeholder values and footer rows are handled automatically.",
        )
        st.caption("Accepts `.csv`, `.xlsx`, `.xls`")
    with status_col:
        if uploaded is None:
            st.markdown(
                '<div class="rca-check">🟢 Required columns found after upload</div>'
                '<div class="rca-check">🟢 Missing values are checked automatically</div>'
                '<div class="rca-check">🟢 Values are checked against expected ranges</div>',
                unsafe_allow_html=True,
            )

if uploaded is None:
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

try:
    peek = pd.ExcelFile(path).sheet_names if suffix != ".csv" else []
except Exception:
    peek = []


# --------------------------------------------------------------------------- #
# 2 · Detected layout  (open it only if something's off)
# --------------------------------------------------------------------------- #
# A first probe with whatever sheet/header the widgets already hold, so the
# expander can decide whether to open itself before its widgets are drawn.
_sheet0 = st.session_state.get("sheet_pick")
_sheet0 = None if _sheet0 in (None, AUTO) else _sheet0
_hdr0 = st.session_state.get("hdr_pick", "")
_hdr0 = int(_hdr0) - 1 if str(_hdr0).strip().isdigit() else None
try:
    pre = probe(file_key, path, _sheet0, _hdr0)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not read the file: {exc}")
    st.stop()

pre_ok = pre.mapping.is_complete and not pre.mapping.ambiguous and not pre.needs_review

with st.container(border=True):
    file_col, check_col = st.columns([1.0, 1.45], gap="large")
    with file_col:
        st.markdown(f"**{uploaded.name}**")
        st.caption(f"{pre.n_rows:,} rows · {len(pre.source_columns)} columns")
        if pre_ok:
            st.success("Valid file", icon="✅")
        else:
            st.warning("Needs review", icon="⚠️")
    with check_col:
        checks = [
            (pre.mapping.is_complete, "Required columns found (date, stage, discharge)"),
            (not pre.needs_review, "No ambiguous mappings in the uploaded data"),
            (True, "Stage and discharge values are ready for validation"),
        ]
        for passed, message in checks:
            icon = "🟢" if passed else "🟠"
            st.markdown(f'<div class="rca-check">{icon} {message}</div>', unsafe_allow_html=True)

with st.expander("Review detected columns" + ("" if pre_ok else "  ⚠️  check this"),
                 expanded=not pre_ok):
    top = st.columns([2, 1])
    if peek:
        sheet_pick = top[0].selectbox("Sheet", [AUTO, *peek], key="sheet_pick")
        sheet = None if sheet_pick == AUTO else sheet_pick
    else:
        sheet = None
    hdr_pick = top[1].text_input("Header row", value="", placeholder="auto", key="hdr_pick")
    header_row = int(hdr_pick) - 1 if hdr_pick.strip().isdigit() else None

    try:
        base_report = probe(file_key, path, sheet, header_row)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the file: {exc}")
        st.stop()

    st.caption("Blank = auto-detect. Set the starred fields if they're wrong.")
    options = [AUTO, *base_report.source_columns]

    for canonical, cands in base_report.mapping.ambiguous.items():
        picked, *others = cands
        st.warning(
            f"**{FIELD_LABELS.get(canonical, canonical)}** matched more than one "
            f"column — using **{picked}**, not {', '.join(others)}. Change it below "
            f"if that's wrong.",
            icon="⚠️",
        )
    conv = [f"{FIELD_LABELS.get(k, k)} converted from {u.label}"
            for k, u in (base_report.units or {}).items()
            if getattr(u, "detected", False) and getattr(u, "factor", 1.0) != 1.0]
    if conv:
        st.caption("Units: " + "; ".join(conv) + ".")

    def _map(field_name: str, col) -> str | None:
        guess = base_report.mapping.fields.get(field_name)
        idx = options.index(guess) if guess in base_report.source_columns else 0
        star = " *" if field_name in REQUIRED_FIELDS else ""
        pick = col.selectbox(FIELD_LABELS[field_name] + star, options, index=idx,
                             key=f"map_{field_name}")
        return pick if pick != AUTO else None

    overrides: dict[str, str] = {}
    req_cols = st.columns(len(REQUIRED_FIELDS))
    for col, field_name in zip(req_cols, REQUIRED_FIELDS):
        chosen = _map(field_name, col)
        if chosen:
            overrides[field_name] = chosen

    optional = [f for f in ALL_FIELDS if f not in REQUIRED_FIELDS]
    if any(base_report.mapping.fields.get(f) for f in optional) or \
            st.checkbox("Map optional columns (quality, field notes, site, uncertainty…)"):
        opt_cols = st.columns(3)
        for i, field_name in enumerate(optional):
            chosen = _map(field_name, opt_cols[i % 3])
            if chosen:
                overrides[field_name] = chosen

    if len(set(overrides.values())) != len(overrides):
        st.warning("The same column is mapped to more than one field.")

try:
    result = validate(file_key, path, sheet, header_row, tuple(sorted(overrides.items())))
except Exception as exc:  # noqa: BLE001
    st.error(
        "Couldn't identify a **date**, a **stage** and a **discharge** column. "
        "Open **Detected layout** above and set the starred fields.\n\n"
        f"```\n{exc}\n```"
    )
    st.stop()

report = result.load_report


# --------------------------------------------------------------------------- #
# 3 · Fit controls
# --------------------------------------------------------------------------- #
st.markdown('<div class="rca-section-label">Stage–discharge fit</div>', unsafe_allow_html=True)
st.markdown('<div class="rca-section-help">Choose the curve shape and uncertainty settings, then review the fitted result below.</div>', unsafe_allow_html=True)

site = None
control_cols = st.columns(3 if result.is_multi_site else 2)
segments = control_cols[0].selectbox(
    "Curve shape", [1, 2, 3, "auto"],
    format_func=lambda n: {1: "Single power law", 2: "2 segments", 3: "3 segments",
                           "auto": "Auto (BIC picks 1–4)"}[n],
    help="A compound control (a low-flow notch under a wider channel) needs more "
         "than one power-law segment.",
)
method_label = control_cols[1].selectbox(
    "Method", ["Least squares", "Bayesian"],
    help="Least squares: fast log–log regression (auto-weighted by a discharge-"
         "uncertainty column). Bayesian: thodson-usgs `ratingcurve` (PyMC) — "
         "samples h₀, slopes and breakpoints jointly; needs the `[bayesian]` "
         "extra and ≈ 1 min for the first fit.",
)
method = "bayesian" if method_label == "Bayesian" else "ols"

if result.is_multi_site:
    site_pick = control_cols[2].selectbox("Site", ["(all sites)", *result.sites])
    site = None if site_pick == "(all sites)" else site_pick

bayesian_sampler = "auto"
if method == "bayesian":
    bayesian_sampler = st.radio(
        "Sampler", ["auto", "nuts", "advi"], horizontal=True,
        format_func=lambda s: {"auto": "auto (NUTS ≤ 200 gaugings)",
                               "nuts": "NUTS — exact, slow",
                               "advi": "ADVI — variational, fast"}[s],
    )

t1, t2 = st.columns(2)
set_h0 = t1.checkbox("Set h₀ (stage of zero flow) by hand",
                     help="Off = estimate it from the low-flow gaugings.")
h0 = t1.number_input("h₀ (m)", value=0.18, step=0.01, format="%.3f") if set_h0 else None

fixed_b = None
if method == "ols":
    if t2.checkbox("Impose the exponent b",
                   help="Pin b from the control type (≈1.5 broad-crested weir, "
                        "≈2–2.5 natural section control, ≈2.5 V-notch) and fit only "
                        "a — for records too sparse or scattered to identify b on "
                        "their own. Single-segment only."):
        fixed_b = t2.number_input("b", min_value=0.1, max_value=5.0, value=2.0,
                                  step=0.1, format="%.2f")
        if segments != 1:
            t2.caption("↳ forced to a single segment.")
            segments = 1

# ---- uncertainty & flags --------------------------------------------------
with st.expander("Uncertainty & point flags"):
    uncertainty_pct = st.number_input(
        "Assumed discharge-measurement uncertainty (±%)", min_value=0.5, max_value=100.0,
        value=float(DEFAULT_DISCHARGE_UNCERTAINTY_PCT), step=0.5,
        help="Applied to gaugings with no value in a mapped 'Discharge uncertainty "
             "(±%)' column. Sets the confidence/prediction band width; a *varying* "
             "mapped column also re-weights the fit point by point.",
    )
    threshold = st.slider(
        "Mark a gauging 'uncertain' in the report once it sits this far off the curve",
        5, 100, int(round(DEFAULT_UNCERTAINTY_THRESHOLD * 100)), 5, format="%d%%",
    ) / 100.0

# ---- advanced -----------------------------------------------------------
with st.expander("Advanced"):
    rating_step = st.number_input(
        "Rating-table step (m)", min_value=0.001, max_value=1.0,
        value=float(DEFAULT_STAGE_STEP_M), step=0.005, format="%.3f",
        help="Stage increment of the stage → discharge lookup table.",
    )

    st.markdown("**Manning cross-section check** *(optional — flood work)*")
    st.caption(
        "The rating is fitted only over the stages you've gauged. To read "
        "discharge at higher stages it must be **extrapolated**, and a power law "
        "can extrapolate badly. Give a surveyed cross-section (offset + bed "
        "elevation) and the water-surface slope: the tool builds an independent "
        "Manning discharge from the channel geometry and flags where the "
        "extrapolated rating disagrees with it. Skip it for a purely low-flow rating."
    )
    sec_file = st.file_uploader("Cross-section CSV (offset + elevation)", type=["csv"], key="xsec")
    m1, m2, m3 = st.columns(3)
    section_slope = m1.number_input("Water-surface slope (m/m)", min_value=0.0, value=0.0,
                                    step=0.0001, format="%.5f")
    section_n = m2.number_input("Manning's n (0 = calibrate)", min_value=0.0, max_value=0.3,
                                value=0.0, step=0.005, format="%.3f")
    section_offset = m3.number_input("Stage → WSE offset (m)", value=0.0, step=0.01, format="%.3f")

    section_csv = None
    if sec_file is not None and section_slope > 0:
        sec_path = Path(tempfile.gettempdir()) / f"rca_xsec_{hashlib.md5(sec_file.getvalue()).hexdigest()}.csv"
        sec_path.write_bytes(sec_file.getvalue())
        section_csv = str(sec_path)
    elif sec_file is not None:
        st.warning("Enter a positive water-surface slope to run the Manning check.")


# --------------------------------------------------------------------------- #
# 4 · Run
# --------------------------------------------------------------------------- #
try:
    with st.spinner("Sampling the posterior… (~1 min on the first Bayesian fit)"
                    if method == "bayesian" else "Fitting…"):
        outcome, fit_df, report_bytes, rating_table, rating_csv = fit_and_report(
            file_key, path, sheet, header_row, tuple(sorted(overrides.items())),
            h0, segments, site, threshold, uncertainty_pct, rating_step, method,
            bayesian_sampler, fixed_b,
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


# --------------------------------------------------------------------------- #
# 5 · Rows used
# --------------------------------------------------------------------------- #
cleaned = result.cleaned

if result.invalid_count:
    with st.expander(f"{result.invalid_count} excluded row(s) — why"):
        cols = [c for c in (DATE, STAGE_M, DISCHARGE_CMS, "validation_notes") if c in cleaned.columns]
        st.dataframe(_friendly(cleaned.loc[~cleaned["is_valid"], cols]),
                     width="stretch", hide_index=True)
if result.warning_count:
    with st.expander(f"{result.warning_count} kept row(s) with a warning"):
        cols = [c for c in (DATE, STAGE_M, DISCHARGE_CMS, "warning_notes") if c in cleaned.columns]
        st.dataframe(_friendly(cleaned.loc[cleaned["has_warning"], cols]),
                     width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# 6 · The rating curve
# --------------------------------------------------------------------------- #
st.markdown('<div class="rca-section-label">Stage–discharge fit</div>', unsafe_allow_html=True)
st.markdown('<div class="rca-section-help">Fitted rating curve with an uncertainty band.</div>', unsafe_allow_html=True)
if not outcome.is_plausible:
    st.error("**Not a plausible rating curve** — see the notes below.")
elif outcome.warnings:
    st.warning("**Fitted, with warnings.**")
else:
    st.success(f"**Rating curve fitted.**   R² = {p['r_squared']:.3f}")

bands = p.get("bands")
pct = int(round(bands["level"] * 100)) if bands else 95
r2_label = "weighted R²" if p.get("weighted") else "R²"
r2_value = p.get("r_squared_weighted") if p.get("weighted") else p["r_squared"]

bits = [f"{p['n_points']} gaugings used"]
if p["h0_estimated"]:
    hd = p.get("h0_diagnostics") or {}
    bits.append("h₀ weakly identified" if hd.get("railed")
                else f"h₀ estimated ({hd.get('method', '?')})")
else:
    bits.append("h₀ set by hand")
if bands and bands.get("b_ci") and not p.get("b_fixed"):
    bits.append(f"b {pct}% CI [{bands['b_ci'][0]:.2f}, {bands['b_ci'][1]:.2f}]")
if bands:
    unit = "posterior draws" if bands.get("kind") == "posterior" else "bootstrap refits"
    bits.append(f"±{bands['ci_halfwidth_pct_at_median']:.0f}% band at mid-stage "
                f"({bands['n_success']} {unit})")
else:
    bits.append("bands need ≥ 4 usable gaugings")
st.caption("  ·  ".join(bits))

if outcome.warnings:
    st.markdown("\n".join(f"- {w}" for w in outcome.warnings))

log_scale = st.toggle("Log–log axes", value=False)
with st.container(border=True):
    plot_col, summary_col = st.columns([3.7, 1.3], gap="large")
    with plot_col:
        st.pyplot(
            make_rating_curve_figure(fit_df, a=p["a"], b=p["b"], h0=p["h0"], log_scale=log_scale, fit=p),
            width="stretch",
        )
    with summary_col:
        st.markdown("#### Curve summary")
        st.code(p["equation"], language="text")
        st.metric("Measurements", p["n_points"])
        st.metric(r2_label, f"{r2_value:.3f}")

with st.expander("How the fit was set up"):
    if p.get("method") == "bayesian":
        bx = p.get("bayes", {})
        st.write(f"**Bayesian** (thodson-usgs `ratingcurve`, PyMC {bx.get('sampler', '?').upper()}). "
                 + (bx.get("auto_segments_note") or ""))
    else:
        st.write("**Least squares** (log–log regression).")
    if p.get("weighted"):
        st.write(f"Weighted by the per-point discharge-uncertainty column "
                 f"(mean ±{p['mean_uncertainty_pct']:.1f}%).")
    elif p.get("uncertainty_source") == "column":
        st.write("A discharge-uncertainty column was found but every value is equal — not re-weighted.")
    else:
        st.write(f"Discharge uncertainty assumed at ±{p['uncertainty_pct_default']:.1f}% "
                 f"for every gauging — not re-weighted.")
    if bands:
        st.write(
            f"{pct}% **confidence** band = how well the mean curve is known; "
            f"{pct}% **prediction** band = where the next gauging would fall. "
            f"Bands span the gauged stage range only, not the extrapolation."
        )
        if bands.get("h0_ci"):
            src = "posterior" if bands.get("kind") == "posterior" else "re-estimated per replicate"
            st.write(f"h₀ {pct}% interval [{bands['h0_ci'][0]:.3f}, {bands['h0_ci'][1]:.3f}] m ({src}).")
        if bands.get("breakpoint_ci"):
            st.write("Breakpoint interval(s): "
                     + "; ".join(f"[{lo:.3f}, {hi:.3f}] m" for lo, hi in bands["breakpoint_ci"]))


# --------------------------------------------------------------------------- #
# 7 · Diagnostics
# --------------------------------------------------------------------------- #
valid_stage = result.cleaned.loc[result.cleaned["is_valid"], STAGE_M]
stage_min = float(valid_stage.min()) if not valid_stage.empty else float("nan")
stage_max = float(valid_stage.max()) if not valid_stage.empty else float("nan")
with st.container(border=True):
    st.markdown('<div class="rca-section-label">Diagnostics</div>', unsafe_allow_html=True)
    st.markdown('<div class="rca-section-help">Key indicators of fit quality and data consistency.</div>', unsafe_allow_html=True)
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Number of measurements", p["n_points"])
    q2.metric("R² (goodness of fit)", f"{r2_value:.3f}")
    q3.metric("Warnings", result.warning_count)
    q4.metric("Stage range (m)", f"{stage_min:.2f} – {stage_max:.2f}")

drift = p.get("drift")
mc = p.get("manning")
if drift or mc:
    st.markdown("#### Quality checks")

if drift:
    if drift["flag"] == "likely":
        st.warning(f"⏳ **Rating shift likely.** {drift['message']}")
    elif drift["flag"] in ("possible", "unassessable"):
        st.info(f"⏳ {drift['message']}")
    else:
        st.success(f"⏳ No temporal drift detected ({drift['date_min']} → {drift['date_max']}).")
    cp = drift.get("changepoint")
    if cp is not None:
        st.caption(
            f"Most likely changepoint **{cp['date']}** — {cp['shift_pct']:+.0f}% across it "
            f"(p={cp['p_value']:.3f}; {cp['n_before']} gaugings before, {cp['n_after']} after)."
        )
    resid_fig = make_residual_time_figure(fit_df, p)
    if resid_fig is not None:
        with st.expander("Residuals over time", expanded=drift["flag"] == "likely"):
            st.pyplot(resid_fig, width="stretch")

if mc:
    if mc.get("flag") in ("diverges", "implausible-n"):
        st.warning(f"📐 {mc['message']}")
    elif mc.get("flag") in ("check", "unusable"):
        st.info(f"📐 {mc['message']}")
    else:
        st.success(f"📐 {mc['message']}")


# --------------------------------------------------------------------------- #
# 8 · Download
# --------------------------------------------------------------------------- #
tag = f"_{site}" if site else ""
with st.container(border=True):
    export_col, table_col = st.columns([1, 1], gap="large")
    with export_col:
        st.markdown('<div class="rca-section-label">Export report</div>', unsafe_allow_html=True)
        st.caption("Generate a report with the data, fitted curve, uncertainty, and diagnostics.")
        st.download_button(
            "⬇︎  Export Excel report", data=report_bytes,
            file_name=f"rating_curve_report{tag}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="content",
        )
    with table_col:
        st.markdown('<div class="rca-section-label">Rating table</div>', unsafe_allow_html=True)
        st.caption(f"Stage → discharge lookup every {rating_step:g} m.")
        st.download_button(
            "⬇︎  Download rating table (CSV)", data=rating_csv,
            file_name=f"rating_table{tag}.csv", mime="text/csv", width="content",
        )
with st.expander(f"Rating table — stage → discharge every {rating_step:g} m ({len(rating_table)} rows)"):
    st.dataframe(rating_table, width="stretch", hide_index=True)
