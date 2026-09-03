"""Streamlit front end for the Rating Curve Automater.

Launch it with ``rca app`` (from any install), which runs ``streamlit run`` on
this file.

Layout convention: **every input lives in the sidebar, every result in the main
pane.** A thin view over :class:`rating_curve_automater.workflow.RatingCurveWorkflow`,
the headless load -> validate -> fit -> export controller.
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
# Empty state
# --------------------------------------------------------------------------- #
st.title("📈 Rating Curve Automater")

uploaded = st.sidebar.file_uploader(
    "Gauging workbook", type=["xlsx", "xls", "csv"],
    help="A table of field gaugings: a date, a stage and a discharge column. "
         "Messy headers, extra sheets and unit labels are handled automatically.",
)

if uploaded is None:
    st.markdown(
        "#### Turn a spreadsheet of field gaugings into a rating curve.\n"
        "Fits `Q = a · (H − h₀)^b` — single or multi-segment — with uncertainty "
        "bands, a stage → discharge table, and temporal-drift checks."
    )
    a, b, c = st.columns(3)
    a.info("**1 · Upload**\n\nDrop your `.xlsx` / `.xls` / `.csv` in the sidebar.")
    b.info("**2 · Check**\n\nConfirm the detected columns and pick the fit settings.")
    c.info("**3 · Download**\n\nGrab the Excel report and the rating table.")
    st.caption(
        "Provisional software, no warranty — review every curve as a qualified "
        "hydrographer before operational use."
    )
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
# Sidebar · data source
# --------------------------------------------------------------------------- #
with st.sidebar:
    try:
        peek = pd.ExcelFile(path).sheet_names if suffix != ".csv" else []
    except Exception:
        peek = []

    with st.expander("Data source", expanded=False):
        if peek:
            sheet_choice = st.selectbox("Sheet", [AUTO, *peek])
            sheet = None if sheet_choice == AUTO else sheet_choice
        else:
            sheet = None
        header_text = st.text_input("Header row (1 = first row)", value="", placeholder="auto-detect")
        header_row = (int(header_text) - 1) if header_text.strip().isdigit() else None

try:
    base_report = probe(file_key, path, sheet, header_row)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not read the file: {exc}")
    st.stop()


# --------------------------------------------------------------------------- #
# Sidebar · columns
# --------------------------------------------------------------------------- #
resolved = [FIELD_LABELS[f] for f in REQUIRED_FIELDS if base_report.mapping.fields.get(f)]
cols_ok = len(resolved) == len(REQUIRED_FIELDS) and not base_report.mapping.ambiguous
with st.sidebar:
    with st.expander(
        "Columns  " + ("✅" if cols_ok else "⚠️"),
        expanded=not cols_ok,
    ):
        if cols_ok:
            st.caption("Auto-detected. Change any that are wrong.")
        else:
            st.caption("Auto-detection is unsure here — set the starred fields.")

        options = [AUTO, *base_report.source_columns]

        def _map(field_name: str) -> str | None:
            guess = base_report.mapping.fields.get(field_name)
            idx = options.index(guess) if guess in base_report.source_columns else 0
            star = " *" if field_name in REQUIRED_FIELDS else ""
            pick = st.selectbox(FIELD_LABELS[field_name] + star, options, index=idx,
                                key=f"map_{field_name}")
            return pick if pick != AUTO else None

        overrides: dict[str, str] = {}
        optional = [f for f in ALL_FIELDS if f not in REQUIRED_FIELDS]
        for field_name in REQUIRED_FIELDS:
            chosen = _map(field_name)
            if chosen:
                overrides[field_name] = chosen
        show_optional = any(base_report.mapping.fields.get(f) for f in optional) or \
            st.checkbox("Optional columns (quality, notes, site…)")
        if show_optional:
            for field_name in optional:
                chosen = _map(field_name)
                if chosen:
                    overrides[field_name] = chosen
        if len(set(overrides.values())) != len(overrides):
            st.warning("The same column is mapped to more than one field.")

try:
    result = validate(file_key, path, sheet, header_row, tuple(sorted(overrides.items())))
except Exception as exc:  # noqa: BLE001
    st.error(
        "Couldn't identify a **date**, **stage** and **discharge** column — "
        "open **Columns** in the sidebar and set the starred fields.\n\n"
        f"```\n{exc}\n```"
    )
    st.stop()

report = result.load_report


# --------------------------------------------------------------------------- #
# Sidebar · fit settings
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.subheader("Fit settings")

    site = None
    if result.is_multi_site:
        site_choice = st.selectbox("Site", ["(all sites)", *result.sites])
        site = None if site_choice == "(all sites)" else site_choice

    h0_mode = st.radio(
        "Stage of zero flow (h₀)", ["Estimate from the gaugings", "Set a value"],
        help="h₀ is the stage at which flow stops. Estimated from the low-flow "
             "gaugings by default; set it by hand if you have a surveyed value.",
    )
    h0 = None
    if h0_mode == "Set a value":
        h0 = st.number_input("h₀ (m)", value=0.18, step=0.01, format="%.3f")

    shape = st.selectbox(
        "Curve shape",
        [1, 2, 3, "auto"],
        format_func=lambda n: {
            1: "Single power law",
            2: "2 segments (compound control)",
            3: "3 segments",
            "auto": "Auto — let BIC choose 1–4",
        }[n],
        help="A compound control (e.g. a low-flow notch under a wider channel) "
             "needs more than one power-law segment.",
    )
    segments: int | str = shape

    # ---- advanced ----------------------------------------------------------
    with st.expander("Advanced", expanded=False):
        method_label = st.radio(
            "Fitting method", ["Least squares", "Bayesian"], horizontal=True,
            help="Least squares: fast log–log regression (auto-weighted when a "
                 "discharge-uncertainty column varies). Bayesian: thodson-usgs "
                 "`ratingcurve` (PyMC) — needs the `[bayesian]` extra; first fit ≈ 1 min.",
        )
        method = "bayesian" if method_label == "Bayesian" else "ols"

        bayesian_sampler = "auto"
        if method == "bayesian":
            bayesian_sampler = st.radio(
                "Sampler", ["auto", "nuts", "advi"], horizontal=True,
                format_func=lambda s: {"auto": "auto", "nuts": "NUTS (exact)",
                                       "advi": "ADVI (fast)"}[s],
            )

        fixed_b = None
        if method == "ols":
            if st.checkbox(
                "Impose exponent b",
                help="Pin the exponent from the control type (≈1.5 weir, ≈2–2.5 "
                     "section control) and fit only a. Use when the gaugings are "
                     "too few / too scattered to identify b. Single-segment only.",
            ):
                fixed_b = st.number_input("b", min_value=0.1, max_value=5.0,
                                          value=2.0, step=0.1, format="%.2f")
                if segments != 1:
                    st.caption("↳ forced to a single segment.")
                    segments = 1

        uncertainty_pct = st.number_input(
            "Assumed discharge uncertainty (±%)",
            min_value=0.5, max_value=100.0, value=float(DEFAULT_DISCHARGE_UNCERTAINTY_PCT), step=0.5,
            help="Used for gaugings with no value in a 'Discharge uncertainty (±%)' "
                 "column. Map that column to weight the fit point-by-point.",
        )
        threshold = st.slider(
            "Flag a gauging when it sits this far off the curve",
            5, 100, int(round(DEFAULT_UNCERTAINTY_THRESHOLD * 100)), 5, format="%d%%",
        ) / 100.0
        rating_step = st.number_input(
            "Rating-table step (m)", min_value=0.001, max_value=1.0,
            value=float(DEFAULT_STAGE_STEP_M), step=0.005, format="%.3f",
        )

        st.markdown("**Manning cross-section check** *(optional)*")
        st.caption(
            "Sanity-checks the curve's extrapolation above the highest gauging "
            "against surveyed channel geometry."
        )
        sec_file = st.file_uploader("Cross-section CSV (offset + elevation)", type=["csv"], key="xsec")
        section_slope = st.number_input("Channel slope (m/m)", min_value=0.0, value=0.0,
                                        step=0.0001, format="%.5f")
        section_n = st.number_input("Manning's n (0 = calibrate)", min_value=0.0, max_value=0.3,
                                    value=0.0, step=0.005, format="%.3f")
        section_offset = st.number_input("Stage → water-surface-elevation offset (m)",
                                         value=0.0, step=0.01, format="%.3f")

        section_csv = None
        if sec_file is not None and section_slope > 0:
            sec_path = Path(tempfile.gettempdir()) / f"rca_xsec_{hashlib.md5(sec_file.getvalue()).hexdigest()}.csv"
            sec_path.write_bytes(sec_file.getvalue())
            section_csv = str(sec_path)
        elif sec_file is not None:
            st.warning("Enter a positive channel slope to run the Manning check.")


# --------------------------------------------------------------------------- #
# Run
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
# Main · data check
# --------------------------------------------------------------------------- #
if site:
    st.caption(f"Site: **{site}**")

cleaned = result.cleaned
used = result.valid_count
total = used + result.invalid_count
d1, d2, d3 = st.columns(3)
d1.metric("Valid rows", used, help=f"{used} of {total} rows go into the fit.")
d2.metric("Invalid (excluded)", result.invalid_count)
d3.metric("Warnings (kept)", result.warning_count)

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

if report.needs_review:
    with st.expander("⚠️ The loader wasn't fully confident — check its choices"):
        st.text(report.describe())
        st.dataframe(_friendly(report.sample), width="stretch", hide_index=True)

st.divider()


# --------------------------------------------------------------------------- #
# Main · the rating curve
# --------------------------------------------------------------------------- #
if not outcome.is_plausible:
    st.error("**Not a plausible rating curve** — see the notes below.")
elif outcome.warnings:
    st.warning("**Fitted, with warnings.**")
else:
    st.success(f"**Rating curve fitted.**   R² = {p['r_squared']:.3f}")

st.code(p["equation"], language="text")

bands = p.get("bands")
pct = int(round(bands["level"] * 100)) if bands else 95
r2_label = "weighted R²" if p.get("weighted") else "R²"
r2_value = p.get("r_squared_weighted") if p.get("weighted") else p["r_squared"]

mcol = st.columns(4)
mcol[0].metric("a", f"{p['a']:.4f}")
mcol[1].metric("b", f"{p['b']:.4f}", "imposed" if p.get("b_fixed") else None, delta_color="off")
mcol[2].metric("h₀ (m)", f"{p['h0']:.3f}")
mcol[3].metric(r2_label, f"{r2_value:.3f}")

# one tidy line of context instead of a stack of metric sub-labels + captions
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
st.pyplot(
    make_rating_curve_figure(fit_df, a=p["a"], b=p["b"], h0=p["h0"], log_scale=log_scale, fit=p),
    width="stretch",
)

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
# Main · diagnostics
# --------------------------------------------------------------------------- #
drift = p.get("drift")
mc = p.get("manning")
if drift or mc:
    st.subheader("Diagnostics")

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
# Main · downloads
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("Download")
tag = f"_{site}" if site else ""
g1, g2 = st.columns(2)
g1.download_button(
    "⬇︎  Excel report", data=report_bytes,
    file_name=f"rating_curve_report{tag}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    width="stretch",
)
g2.download_button(
    "⬇︎  Rating table (CSV)", data=rating_csv,
    file_name=f"rating_table{tag}.csv", mime="text/csv", width="stretch",
)
with st.expander(f"Rating table — stage → discharge every {rating_step:g} m ({len(rating_table)} rows)"):
    st.dataframe(rating_table, width="stretch", hide_index=True)
