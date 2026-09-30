"""Streamlit front end for the Rating Curve Automater.

Launch it with ``rca app`` (from any install), which runs ``streamlit run`` on
this file.

The interface is a one-screen dashboard: a narrow control panel on the left
(upload, column mapping, fit settings) and the result on the right (status and
downloads, headline numbers, the plot beside the quality checks, then tabs for
the detail). It is a thin view over
:class:`rating_curve_automater.workflow.RatingCurveWorkflow`.
"""

from __future__ import annotations

import hashlib
import html
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

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
        --rca-text: #eef5ff;
        --rca-muted: #9eb0c4;
        --rca-blue: #61a8ff;
        --rca-green: #45d39a;
        --rca-amber: #f4b860;
        --rca-red: #ff8a80;
        --rca-focus: #9ccbff;
    }
    .stApp { background: var(--rca-bg); color: var(--rca-text); }
    .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp [data-testid="stMetricLabel"], .stApp [data-testid="stMetricValue"], .stApp [data-testid="stMetricDelta"] {
        font-family:system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
    }
    header[data-testid="stHeader"] { background: transparent; }
    [data-testid="stAppDeployButton"] { display: none; }
    .block-container { max-width: 1680px; padding: .8rem 1.25rem 1.5rem; }
    [data-testid="stVerticalBlock"] { gap: .45rem; }
    [data-testid="stHorizontalBlock"] { gap: .6rem; }
    p, label, [data-testid="stMarkdownContainer"] { color: #c6d2e2; }
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { font-size: .78rem; color: var(--rca-muted); }
    [data-testid="stWidgetLabel"] p { font-size: .82rem; }
    /* Streamlit pulls markdown up by 1rem to cancel a <p> margin; our HTML blocks have none */
    [data-testid="stMarkdownContainer"]:has(> .rca-title, > .rca-label, > .rca-file, > .rca-eq, > .rca-empty) { margin-bottom: 0; }

    /* title bar — leaves room for Streamlit's menu on the right */
    .rca-title { display:flex; align-items:baseline; justify-content:center; flex-wrap:wrap; gap:.2rem .75rem; margin:0 0 .6rem; text-align:center; }
    .rca-title h1 { font-size:1.75rem !important; line-height:1.2 !important; font-weight:800 !important; margin:0 !important; padding:0 !important; letter-spacing:-.03em; }

    /* panels */
    .st-key-rca-panel, .st-key-rca-plot, .st-key-rca-checks, .st-key-rca-colmap-inline {
        background: var(--rca-panel); border-color: var(--rca-border) !important; border-radius: 8px;
    }
    .st-key-rca-panel { gap: .5rem; }
    .rca-label { color:var(--rca-muted); font-size:.7rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; margin:.3rem 0 -.2rem; }
    .rca-file { color:var(--rca-muted); font-size:.78rem; }

    /* file uploader: inviting without taking over the working surface */
    [data-testid="stFileUploaderDropzone"] { display:flex; flex-direction:column; align-items:center; justify-content:center; background:var(--rca-panel-2); border:1px dashed #6f8dab; border-radius:8px; padding:.7rem; transition:border-color .15s ease, background .15s ease; }
    [data-testid="stFileUploaderDropzone"]:hover { background:#223246; border-color:var(--rca-blue); }
    [data-testid="stFileUploaderDropzoneInstructions"] { display:none; }

    /* first-run welcome card */
    .rca-welcome { max-width:55rem; margin:.8rem auto 0; padding:1.5rem 1.6rem 1.35rem; background:linear-gradient(135deg, #1a2a3b 0%, #172230 72%); border:1px solid var(--rca-border); border-radius:12px; box-shadow:0 14px 30px rgba(0,0,0,.14); }
    .rca-welcome-kicker { color:var(--rca-blue); font-size:.72rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; margin-bottom:.45rem; }
    .rca-welcome h2 { color:var(--rca-text); font-size:1.45rem; line-height:1.2; margin:0 0 .45rem; letter-spacing:-.02em; }
    .rca-welcome p { margin:.2rem 0; color:#c6d2e2; font-size:.95rem; line-height:1.45; }
    .rca-welcome-steps { display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:.7rem; margin-top:1.15rem; }
    .rca-welcome-step { padding:.7rem .75rem; background:rgba(29,42,57,.72); border:1px solid #2d4054; border-radius:8px; }
    .rca-welcome-step b { display:block; color:#eef5ff; font-size:.85rem; margin-bottom:.2rem; text-align:center; }
    .rca-welcome-step span { display:block; color:var(--rca-muted); font-size:.78rem; line-height:1.35; text-align:center; }
    .st-key-rca-upload-below { max-width:55rem; margin:.9rem auto 0; background:var(--rca-panel); border-color:var(--rca-border) !important; border-radius:12px; }
    [role="tooltip"], [data-baseweb="tooltip"] { max-width:27rem !important; width:27rem !important; font-size:.88rem !important; line-height:1.45 !important; }
    @media (max-width: 760px) { .rca-welcome-steps { grid-template-columns:1fr; } }

    /* buttons: one shared height and rhythm for a coherent control row */
    .stButton button, .stDownloadButton button, [data-testid="stPopoverButton"] {
        height: 2.35rem; min-height: 2.35rem; padding: .35rem .8rem; border-radius: 7px; font-size: .85rem; font-weight: 600; line-height:1.2;
        display:inline-flex; align-items:center; justify-content:center;
    }
    [data-testid="stBaseButton-primary"] { background:#2478e5; border:1px solid #3d8bf0; color:#fff; }
    [data-testid="stBaseButton-primary"]:hover { background:#3188f4; border-color:#7eb9ff; color:#fff; }
    [data-testid="stBaseButton-secondary"], [data-testid="stPopoverButton"] { background:var(--rca-panel-2); border:1px solid #3a5570; color:#dbe6f3; }
    [data-testid="stBaseButton-secondary"]:hover, [data-testid="stPopoverButton"]:hover { border-color:var(--rca-blue); color:#fff; }
    .stButton button p, .stDownloadButton button p, [data-testid="stPopoverButton"] p { font-size: inherit; }
    [data-testid="stPopoverButton"] > div { margin-right: 0 !important; }  /* else the label ellipsises */
    .stDownloadButton, .stButton, [data-testid="stPopover"] { align-self:stretch; }
    .stDownloadButton button, .stButton button, [data-testid="stPopoverButton"] { width:100%; }
    button:focus-visible, [role="tab"]:focus-visible, input:focus-visible { outline: 2px solid var(--rca-focus) !important; outline-offset: 2px; }

    /* status line + alerts: one line, not a slab */
    [data-testid="stAlertContainer"] { display:flex; align-items:center; min-height:2.8rem; box-sizing:border-box; padding: .35rem .8rem; border-radius: 6px; }
    [data-testid="stAlertContainer"] p { font-size: .9rem; margin:0; }
    [data-testid="stAlertContainer"] > div { align-items:center; }

    /* headline numbers */
    [data-testid="stMetric"] { display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:7.1rem; height:7.1rem; box-sizing:border-box; background:var(--rca-panel); border:1px solid var(--rca-border); border-radius:6px; padding:.55rem .6rem; text-align:center; }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"] { width:100%; text-align:center; }
    [data-testid="stMetricLabel"] p { font-size:.9rem; color:var(--rca-muted); }
    [data-testid="stMetricValue"] { font-size:1.35rem; line-height:1.3; font-weight:600; color:#f1f6fd; }
    [data-testid="stMetricDelta"] { font-size:.72rem; max-width:100%; }
    [data-testid="stMetricDelta"] p, [data-testid="stMetricLabel"] p { white-space:normal; overflow:visible; }
    [data-testid="stMetricLabel"] p { overflow-wrap:normal; word-break:normal; }
    [data-testid="stTooltipIcon"] { width:1rem !important; height:1rem !important; }
    [data-testid="stTooltipIcon"] svg { width:.8rem !important; height:.8rem !important; }
    .st-key-rca-tiles > div { flex: 1 1 7.5rem; min-width: 7.5rem; }
    .rca-eq { display:flex; gap:.7rem; align-items:center; margin:.35rem 0 .55rem; font-size:1rem; color:var(--rca-muted); }
    .rca-eq code { color:#eef5ff; background:var(--rca-panel-2); border:1px solid var(--rca-border); border-radius:6px; padding:.28rem .65rem; font-size:1.08rem; font-weight:600; white-space:normal; }

    /* quality checks: icon + title + detail, never colour alone */
    .rca-chk { display:grid; grid-template-columns: 1.25rem 1fr; gap:.45rem; padding:.42rem 0; border-top:1px solid var(--rca-border); font-size:.8rem; line-height:1.35; color:#c6d2e2; }
    .rca-chk:first-of-type { border-top:0; }
    .rca-chk b { display:block; color:#eef5ff; font-weight:600; font-size:.84rem; }
    .rca-chk-ic { width:1.25rem; height:1.25rem; border-radius:50%; display:grid; place-items:center; font-size:.72rem; font-weight:800; color:#0d141d; }
    .rca-ok .rca-chk-ic { background:var(--rca-green); }
    .rca-warn .rca-chk-ic { background:var(--rca-amber); }
    .rca-bad .rca-chk-ic { background:var(--rca-red); }
    .rca-info .rca-chk-ic { background:var(--rca-blue); }
    .rca-empty { max-width: 40rem; margin-top: .4rem; }

    /* tabs + expanders */
    [data-testid="stTabs"] [role="tab"] p { font-size:.86rem; }
    [data-testid="stExpander"] details { border-color:var(--rca-border); background:rgba(18,28,40,.65); border-radius:6px; }
    [data-testid="stExpander"] summary { padding:.4rem .65rem; font-size:.86rem; }
    hr { border-color:var(--rca-border); }
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


def _column_concerns(rep) -> list[tuple[str, str]]:
    """``(title, detail)`` for each loader choice a human should double-check.
    Drives the Columns button label, the notes inside it and the Checks list,
    so all three always agree."""
    concerns = []
    if not rep.sheet_confident:
        others = [s for s in rep.available_sheets if s != rep.sheet_name]
        more = f" +{len(others) - 3} more" if len(others) > 3 else ""
        concerns.append((f"Sheet “{rep.sheet_name}” is a best guess",
                         f"Other sheets: {', '.join(others[:3])}{more}." if others else ""))
    if not rep.header_confident:
        concerns.append((f"Header row {rep.header_row + 1} is a best guess", ""))
    for canonical, (picked, *others) in rep.mapping.ambiguous.items():
        concerns.append((f"{FIELD_LABELS.get(canonical, canonical)} matched {len(others) + 1} columns",
                         f"Using {picked}, not {', '.join(others)}."))
    return concerns


_CHECK_ICON = {"ok": "✓", "warn": "!", "bad": "✕", "info": "i"}


def _check(level: str, title: str, detail: str = "") -> str:
    """One row of the checks list. The title carries the verdict in words, so
    the coloured icon is never the only signal."""
    body = f"<b>{html.escape(title)}</b>" + (html.escape(detail) if detail else "")
    return (f'<div class="rca-chk rca-{level}"><span class="rca-chk-ic" aria-hidden="true">'
            f'{_CHECK_ICON[level]}</span><div>{body}</div></div>')


def _display_equation(params: dict) -> str:
    """Readable UI equation; the fitted/report value keeps full precision."""
    if params.get("is_segmented") and params.get("segments"):
        h0 = float(params["h0"])
        return "; ".join(
            f"Q = {float(seg['a']):.3f} · (H − {h0:.3f})^{float(seg['b']):.3f}"
            for seg in params["segments"]
        )
    return f"Q = {float(params['a']):.3f} · (H − {float(params['h0']):.3f})^{float(params['b']):.3f}"


# --------------------------------------------------------------------------- #
# Layout: title bar, then a narrow control panel beside the result area.
# --------------------------------------------------------------------------- #
st.markdown(
    '<div class="rca-title"><h1>Rating Curve Automater</h1></div>',
    unsafe_allow_html=True,
)

def _measurement_uploader():
    return st.file_uploader(
        "Measurement spreadsheet",
        type=["xlsx", "xls", "csv"],
        key="measurement_upload",
        help="Accepted: XLSX, XLS, or CSV up to 200MB. We find date, stage, and discharge columns, then clean invalid rows and detect units before fitting.",
    )


# Keep the first-run action directly below the welcome card. Once a file is
# loaded, move the same uploader back into the compact controls column.
show_side_upload = bool(st.session_state.get("measurement_upload"))
if show_side_upload:
    side, main = st.columns([1, 3.3], gap="medium")
else:
    side = None
    main = st.container()

if show_side_upload:
    panel = side.container(border=True, key="rca-panel")
    with panel:
        uploaded = _measurement_uploader()
else:
    uploaded = None

if uploaded is None:
    with main:
        st.markdown(
            '<div class="rca-welcome">'
            '<div class="rca-welcome-kicker">Ready when you are</div>'
            '<h2>Turn gauging data into a rating curve</h2>'
            '<p>Upload an Excel or CSV spreadsheet to clean, check, fit, and review your stage–discharge relationship.</p>'
            '<div class="rca-welcome-steps">'
            '<div class="rca-welcome-step"><b>1 · Add your file</b><span>Drop it in the upload area below.</span></div>'
            '<div class="rca-welcome-step"><b>2 · Review the fit</b><span>Columns and row quality are checked before fitting.</span></div>'
            '<div class="rca-welcome-step"><b>3 · Export results</b><span>Download the curve report and stage table.</span></div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        with st.container(border=True, key="rca-upload-below"):
            uploaded = _measurement_uploader()
    if uploaded is None:
        st.session_state.pop("file_key", None)
        st.stop()

# If a file was just selected in the first-run uploader, create the controls
# panel for the rest of this run; on the next rerun the uploader moves there.
if not show_side_upload:
    panel = st.container(border=True, key="rca-panel")

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
# Column mapping — a popover when detection worked, inline when it didn't
# --------------------------------------------------------------------------- #
# A first probe with whatever sheet/header the widgets already hold, so we know
# where to draw the mapping widgets before they exist.
_sheet0 = st.session_state.get("sheet_pick")
_sheet0 = None if _sheet0 in (None, AUTO) else _sheet0
_hdr0 = st.session_state.get("hdr_pick", "")
_hdr0 = int(_hdr0) - 1 if str(_hdr0).strip().isdigit() else None
try:
    pre = probe(file_key, path, _sheet0, _hdr0)
except Exception as exc:  # noqa: BLE001
    main.error(f"Could not read the file: {exc}")
    st.stop()

col_concerns = _column_concerns(pre)
alert_box = main.container()
with panel:
    st.markdown(
        f'<div class="rca-file">{pre.n_rows:,} rows · {len(pre.source_columns)} columns</div>',
        unsafe_allow_html=True,
    )
    if pre.mapping.is_complete:
        map_box = st.popover(
            "Columns — check" if col_concerns else "Columns",
            icon=":material/warning:" if col_concerns else ":material/table_view:",
            width="stretch",
            help="Sheet, header row, and which column holds each field.",
        )
    else:
        map_box = main.container(border=True, key="rca-colmap-inline")

with map_box:
    if not pre.mapping.is_complete:
        st.markdown("**Match your columns** — auto-detection couldn't find all three required fields.")
    top = st.columns([2.4, 1], gap="small")
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

    for title, detail in _column_concerns(base_report):
        st.caption(f"⚠️ **{title}**" + (f" — {detail}" if detail else ""))
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
    req_cols = st.columns(len(REQUIRED_FIELDS), gap="small")
    for col, field_name in zip(req_cols, REQUIRED_FIELDS):
        chosen = _map(field_name, col)
        if chosen:
            overrides[field_name] = chosen

    optional = [f for f in ALL_FIELDS if f not in REQUIRED_FIELDS]
    if any(base_report.mapping.fields.get(f) for f in optional) or \
            st.checkbox("Map optional columns (quality, field notes, site, uncertainty…)"):
        opt_cols = st.columns(3, gap="small")
        for i, field_name in enumerate(optional):
            chosen = _map(field_name, opt_cols[i % 3])
            if chosen:
                overrides[field_name] = chosen

    if len(set(overrides.values())) != len(overrides):
        st.warning("The same column is mapped to more than one field.")

try:
    result = validate(file_key, path, sheet, header_row, tuple(sorted(overrides.items())))
except Exception as exc:  # noqa: BLE001
    where = "under **Columns** in the left panel" if pre.mapping.is_complete else "below"
    alert_box.error(
        "Couldn't identify a **date**, a **stage** and a **discharge** column. "
        f"Set the starred fields {where}.\n\n```\n{exc}\n```"
    )
    st.stop()

report = result.load_report


# --------------------------------------------------------------------------- #
# Fit controls (left panel)
# --------------------------------------------------------------------------- #
site = None
bayesian_sampler = "auto"
with panel:
    st.markdown('<div class="rca-label">Fit</div>', unsafe_allow_html=True)
    segments = st.selectbox(
        "Curve shape", [1, 2, 3, "auto"],
        format_func=lambda n: {1: "Single power law", 2: "2 segments", 3: "3 segments",
                               "auto": "Auto (BIC picks 1–4)"}[n],
        help="A compound control (a low-flow notch under a wider channel) needs more "
             "than one power-law segment.",
    )
    method_label = st.selectbox(
        "Method", ["Least squares", "Bayesian"],
        help="Least squares: fast log–log regression (auto-weighted by a discharge-"
             "uncertainty column). Bayesian: thodson-usgs `ratingcurve` (PyMC) — "
             "samples h₀, slopes and breakpoints jointly; needs the `[bayesian]` "
             "extra and ≈ 1 min for the first fit.",
    )
    method = "bayesian" if method_label == "Bayesian" else "ols"

    if result.is_multi_site:
        site_pick = st.selectbox("Site", ["(all sites)", *result.sites])
        site = None if site_pick == "(all sites)" else site_pick

    if method == "bayesian":
        bayesian_sampler = st.selectbox(
            "Sampler", ["auto", "nuts", "advi"],
            format_func=lambda s: {"auto": "Auto (NUTS ≤ 200 gaugings)",
                                   "nuts": "NUTS — exact, slow",
                                   "advi": "ADVI — variational, fast"}[s],
        )

    set_h0 = st.checkbox("Set h₀ by hand",
                         help="h₀ is the stage of zero flow. Off = estimate it from the "
                              "low-flow gaugings.")
    h0 = st.number_input("h₀ (m)", value=0.18, step=0.01, format="%.3f") if set_h0 else None

    fixed_b = None
    if method == "ols":
        if st.checkbox("Impose the exponent b",
                       help="Pin b from the control type (≈1.5 broad-crested weir, "
                            "≈2–2.5 natural section control, ≈2.5 V-notch) and fit only "
                            "a — for records too sparse or scattered to identify b on "
                            "their own. Single-segment only."):
            fixed_b = st.number_input("b", min_value=0.1, max_value=5.0, value=2.0,
                                      step=0.1, format="%.2f")
            if segments != 1:
                st.caption("↳ forced to a single segment.")
                segments = 1

    with st.container(horizontal=True, gap="small"):
        with st.popover("Uncertainty", icon=":material/tune:",
                        help="Measurement uncertainty and the point-flag threshold"):
            uncertainty_pct = st.number_input(
                "Assumed discharge-measurement uncertainty (±%)", min_value=0.5, max_value=100.0,
                value=float(DEFAULT_DISCHARGE_UNCERTAINTY_PCT), step=0.5,
                help="Applied to gaugings with no value in a mapped 'Discharge uncertainty "
                "(±%)' column. Sets the confidence/prediction band width; a *varying* "
                "mapped column also re-weights the fit point by point.",
            )
            threshold = st.slider(
                "Flag a gauging in the report once it sits this far off the curve",
                5, 100, int(round(DEFAULT_UNCERTAINTY_THRESHOLD * 100)), 5, format="%d%%",
            ) / 100.0
        with st.popover("Advanced", icon=":material/more_horiz:",
                        help="Rating-table step and the optional Manning check"):
            rating_step = st.number_input(
                "Rating-table step (m)", min_value=0.001, max_value=1.0,
                value=float(DEFAULT_STAGE_STEP_M), step=0.005, format="%.3f",
                help="Stage increment of the stage → discharge lookup table.",
            )
            st.markdown("**Manning cross-section check** *(optional — flood work)*")
            st.caption("Checks the extrapolation against a surveyed cross-section and "
                       "water-surface slope.")
            sec_file = st.file_uploader("Cross-section CSV (offset + elevation)", type=["csv"], key="xsec")
            m1, m2, m3 = st.columns(3, gap="small")
            section_slope = m1.number_input("Slope (m/m)", min_value=0.0, value=0.0,
                                            step=0.0001, format="%.5f")
            section_n = m2.number_input("Manning n", min_value=0.0, max_value=0.3,
                                        value=0.0, step=0.005, format="%.3f")
            section_offset = m3.number_input("WSE offset (m)", value=0.0, step=0.01, format="%.3f")

            section_csv = None
            if sec_file is not None and section_slope > 0:
                sec_path = Path(tempfile.gettempdir()) / f"rca_xsec_{hashlib.md5(sec_file.getvalue()).hexdigest()}.csv"
                sec_path.write_bytes(sec_file.getvalue())
                section_csv = str(sec_path)
            elif sec_file is not None:
                st.warning("Enter a positive water-surface slope to run the Manning check.")

    st.markdown('<div class="rca-label">View</div>', unsafe_allow_html=True)
    log_scale = st.toggle("Log–log axes", value=False)


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
try:
    with main, st.spinner("Sampling the posterior… (~1 min on the first Bayesian fit)"
                          if method == "bayesian" else "Fitting…"):
        outcome, fit_df, report_bytes, rating_table, rating_csv = fit_and_report(
            file_key, path, sheet, header_row, tuple(sorted(overrides.items())),
            h0, segments, site, threshold, uncertainty_pct, rating_step, method,
            bayesian_sampler, fixed_b,
            section_csv, float(section_slope or 0.0),
            (section_n or None), float(section_offset or 0.0),
        )
except ImportError as exc:
    alert_box.error(str(exc))
    st.stop()
except Exception as exc:  # noqa: BLE001
    alert_box.error(f"Fit failed: {exc}")
    st.stop()

p = outcome.params
cleaned = result.cleaned
bands = p.get("bands")
pct = int(round(bands["level"] * 100)) if bands else 95
r2_label = "weighted R²" if p.get("weighted") else "R²"
r2_value = p.get("r_squared_weighted") if p.get("weighted") else p["r_squared"]
hd = p.get("h0_diagnostics") or {}
drift = p.get("drift")
mc = p.get("manning")
tag = f"_{site}" if site else ""


# --------------------------------------------------------------------------- #
# Status line + downloads
# --------------------------------------------------------------------------- #
with main:
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        if not outcome.is_plausible:
            st.error("**Not a plausible rating curve** — see the checks beside the plot.",
                     icon=":material/error:")
        elif outcome.warnings:
            st.warning(f"**Fitted, with warnings** · {r2_label} = {r2_value:.3f} "
                       f"from {p['n_points']} gaugings", icon=":material/warning:")
        else:
            st.success(f"**Rating curve fitted** · {r2_label} = {r2_value:.3f} "
                       f"from {p['n_points']} gaugings", icon=":material/check_circle:")
        st.download_button(
            "Excel report", data=report_bytes, type="primary", icon=":material/download:",
            file_name=f"rating_curve_report{tag}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Data, fitted curve, uncertainty, diagnostics and the rating table in one workbook.",
        )
        st.download_button(
            "Rating table", data=rating_csv, icon=":material/table:",
            file_name=f"rating_table{tag}.csv", mime="text/csv",
            help=f"Stage → discharge lookup every {rating_step:g} m, as CSV.",
        )

    st.markdown(f'<div class="rca-eq"><span>Equation</span><code>{html.escape(_display_equation(p))}</code></div>',
                unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Headline numbers
# --------------------------------------------------------------------------- #
with main:
    valid_stage = cleaned.loc[cleaned["is_valid"], STAGE_M]
    stage_min = float(valid_stage.min()) if not valid_stage.empty else float("nan")
    stage_max = float(valid_stage.max()) if not valid_stage.empty else float("nan")
    quiet = dict(delta_color="off", delta_arrow="off")

    # Wraps to a second row rather than squeezing when the window is narrow.
    tiles = st.container(horizontal=True, wrap=True, gap="small", key="rca-tiles")
    if p.get("is_segmented"):
        breaks = ", ".join(f"{x:.3f}" for x in p.get("breakpoints") or [])
        tiles.metric("Segments", p["n_segments"], delta=f"breaks at {breaks} m" if breaks else None,
                       help="Per-segment a and b are listed under Fit details.", **quiet)
    else:
        tiles.metric("a", f"{p['a']:.4f}", help="Coefficient in Q = a·(H − h₀)^b.")
        if p.get("b_fixed"):
            b_note = "imposed"
        elif bands and bands.get("b_ci"):
            b_note = f"{pct}% CI {bands['b_ci'][0]:.2f}–{bands['b_ci'][1]:.2f}"
        else:
            b_note = None
        tiles.metric("b", f"{p['b']:.3f}", delta=b_note, help="Exponent in Q = a·(H − h₀)^b.", **quiet)
    if not p["h0_estimated"]:
        h0_note, h0_help = "set by hand", "Stage of zero flow, set by hand."
    else:
        h0_note = "weakly identified" if hd.get("railed") else "estimated"
        h0_help = f"Stage of zero flow, estimated from the low-flow gaugings ({hd.get('method', '?')} method)."
    tiles.metric("h₀ (m)", f"{p['h0']:.3f}", delta=h0_note,
                   delta_color="orange" if hd.get("railed") else "off", delta_arrow="off",
                   help=h0_help)
    tiles.metric(r2_label, f"{r2_value:.3f}")
    tiles.metric("Valid rows", result.valid_count,
                   delta=f"{result.invalid_count} excluded" if result.invalid_count else None,
                   delta_color="orange" if result.invalid_count else "off", delta_arrow="off")
    tiles.metric("Warnings (kept)", result.warning_count,
                   help="Rows kept in the fit but flagged — see the Row warnings tab.")
    if bands:
        unit = "draws" if bands.get("kind") == "posterior" else "refits"
        tiles.metric("Band at mid-stage", f"±{bands['ci_halfwidth_pct_at_median']:.0f}%",
                       delta=f"{pct}% CI · {bands['n_success']} {unit}", **quiet)
    else:
        tiles.metric("Band at mid-stage", "—", delta="needs ≥ 4 gaugings", **quiet)
    tiles.metric("Stage range (m)", f"{stage_min:.2f}–{stage_max:.2f}")


# --------------------------------------------------------------------------- #
# Plot beside the quality checks
# --------------------------------------------------------------------------- #
with main:
    plot_col, check_col = st.columns([2.6, 1], gap="small")
    with plot_col, st.container(border=True, key="rca-plot"):
        st.pyplot(
            make_rating_curve_figure(fit_df, a=p["a"], b=p["b"], h0=p["h0"], log_scale=log_scale,
                                     fit=p, figure=Figure(figsize=(8.4, 5.2), dpi=130)),
            width="stretch",
        )

    checks: list[str] = []
    for title, detail in col_concerns:
        checks.append(_check("warn", title, f"{detail} Change it under Columns.".strip()))
    if not col_concerns:
        checks.append(_check("ok", "Columns mapped", "Date, stage and discharge found."))

    if result.invalid_count:
        checks.append(_check("warn", f"{result.invalid_count} row(s) excluded",
                             "Reasons are in the Excluded rows tab."))
    else:
        checks.append(_check("ok", f"All {result.valid_count} rows usable"))
    if result.warning_count:
        checks.append(_check("info", f"{result.warning_count} row(s) kept with a warning",
                             "Drawn as orange squares; details in the Row warnings tab."))

    level = "bad" if not outcome.is_plausible else "warn"
    for w in outcome.warnings:
        checks.append(_check(level, "Fit warning", w))
    if not outcome.warnings:
        checks.append(_check("ok", "No fit warnings"))

    if hd.get("railed"):
        checks.append(_check("warn", "h₀ weakly identified",
                             "Few low-flow gaugings pin it down — consider setting it by hand."))

    if drift:
        if drift["flag"] == "likely":
            checks.append(_check("warn", "Rating shift likely", drift["message"]))
        elif drift["flag"] in ("possible", "unassessable"):
            checks.append(_check("info", "Drift: " + drift["flag"], drift["message"]))
        else:
            checks.append(_check("ok", "No temporal drift", f"{drift['date_min']} → {drift['date_max']}"))
        cp = drift.get("changepoint")
        if cp is not None:
            checks.append(_check("info", f"Most likely changepoint {cp['date']}",
                                 f"{cp['shift_pct']:+.0f}% across it (p={cp['p_value']:.3f}; "
                                 f"{cp['n_before']} before, {cp['n_after']} after)."))

    if mc:
        if mc.get("flag") in ("diverges", "implausible-n"):
            checks.append(_check("warn", "Manning check", mc["message"]))
        elif mc.get("flag") in ("check", "unusable"):
            checks.append(_check("info", "Manning check", mc["message"]))
        else:
            checks.append(_check("ok", "Manning check", mc["message"]))

    with check_col, st.container(border=True, key="rca-checks"):
        st.markdown('<div class="rca-label">Checks</div>' + "".join(checks), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Detail tabs
# --------------------------------------------------------------------------- #
with main:
    tab_names = ["Rating table", "Residuals over time", "Fit details"]
    if result.invalid_count:
        tab_names.append(f"Excluded rows ({result.invalid_count})")
    if result.warning_count:
        tab_names.append(f"Row warnings ({result.warning_count})")
    tabs = dict(zip(tab_names, st.tabs(
        tab_names, default="Residuals over time" if drift and drift["flag"] == "likely" else None,
    )))

    with tabs["Rating table"]:
        st.caption(f"Stage → discharge every {rating_step:g} m · {len(rating_table)} rows. "
                   "Change the step under Advanced.")
        st.dataframe(rating_table, width="stretch", height=300, hide_index=True)

    with tabs["Residuals over time"]:
        resid_fig = make_residual_time_figure(fit_df, p, figure=Figure(figsize=(10, 3.0), dpi=110))
        if resid_fig is not None:
            st.pyplot(resid_fig, width="stretch")
        else:
            st.caption("The gaugings carry no usable dates, so residuals can't be plotted over time.")

    with tabs["Fit details"]:
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
        if p.get("is_segmented") and p.get("segments"):
            seg = pd.DataFrame(p["segments"]).rename(columns={
                "a": "a", "b": "b", "r_squared": "R²", "n_points": "Gaugings",
                "stage_min": "From stage (m)", "stage_max": "To stage (m)"})
            seg.index = [f"Segment {i + 1}" for i in range(len(seg))]
            st.dataframe(seg, width="content")

    if result.invalid_count:
        with tabs[f"Excluded rows ({result.invalid_count})"]:
            cols = [c for c in (DATE, STAGE_M, DISCHARGE_CMS, "validation_notes") if c in cleaned.columns]
            st.dataframe(_friendly(cleaned.loc[~cleaned["is_valid"], cols]),
                         width="stretch", height=300, hide_index=True)
    if result.warning_count:
        with tabs[f"Row warnings ({result.warning_count})"]:
            cols = [c for c in (DATE, STAGE_M, DISCHARGE_CMS, "warning_notes") if c in cleaned.columns]
            st.dataframe(_friendly(cleaned.loc[cleaned["has_warning"], cols]),
                         width="stretch", height=300, hide_index=True)
