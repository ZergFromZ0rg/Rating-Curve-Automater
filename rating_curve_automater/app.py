"""Streamlit front end for the Rating Curve Automater.

Launch it with ``rca app`` (from any install), which runs ``streamlit run`` on
this file.

The interface has a three-column overview: fit controls on the left, the
curve in the center, and a summary table, file controls and exports on the
right. A second screen holds the detailed results and quality checks. It is a thin view over
:class:`rating_curve_automater.workflow.RatingCurveWorkflow`.
"""

from __future__ import annotations

import hashlib
import html
import tempfile
from io import StringIO
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
        --rca-font: "Helvetica Neue", Helvetica, Arial, sans-serif;
        --rca-bg: #eaf4fb;
        --rca-panel: #ffffff;
        --rca-panel-2: #f4f9fd;
        --rca-border: #bfd4e5;
        --rca-text: #17324a;
        --rca-muted: #5d7489;
        --rca-blue: #2478e5;
        --rca-green: #16855a;
        --rca-amber: #a86511;
        --rca-red: #c74444;
        --rca-focus: #1c6ed0;
    }
    .stApp {
        background-color:var(--rca-bg); color:var(--rca-text);
        background-image:
            radial-gradient(ellipse at 12% 18%, rgba(36,120,229,.035), transparent 27%),
            radial-gradient(ellipse at 83% 11%, rgba(36,120,229,.025), transparent 22%),
            radial-gradient(ellipse at 71% 76%, rgba(36,120,229,.030), transparent 31%),
            radial-gradient(circle, rgba(36,120,229,.040) 0 1px, transparent 1.4px),
            radial-gradient(circle, rgba(69,145,224,.028) 0 1px, transparent 1.5px);
        background-size:auto, auto, auto, 137px 149px, 211px 173px;
        background-position:center, center, center, 19px 31px, 83px 57px;
        background-attachment:fixed;
    }
    /* Include portaled menus and popovers; leave icon and code fonts intact. */
    body, div, p, label, button, input, textarea, select,
    h1, h2, h3, h4, h5, h6, table, th, td {
        font-family: var(--rca-font) !important;
    }
    [data-testid="stMetricValue"], input[type="number"] {
        font-variant-numeric: tabular-nums;
    }
    header[data-testid="stHeader"] { background: transparent; }
    [data-testid="stAppDeployButton"] { display: none; }
    .block-container { max-width: 1680px; padding: .8rem 1.25rem 1.5rem; }
    [data-testid="stVerticalBlock"] { gap: .45rem; }
    [data-testid="stHorizontalBlock"] { gap: .6rem; }
    p, label, [data-testid="stMarkdownContainer"] { color: #314b61; }
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { font-size: .78rem; color: var(--rca-muted); }
    [data-testid="stWidgetLabel"] p { font-size: .82rem; }
    /* Streamlit pulls markdown up by 1rem to cancel a <p> margin; our HTML blocks have none */
    [data-testid="stMarkdownContainer"]:has(> .rca-title, > .rca-label, > .rca-file, > .rca-eq, > .rca-empty) { margin-bottom: 0; }

    /* title bar — leaves room for Streamlit's menu on the right */
    .rca-title { display:flex; flex-direction:column; align-items:center; gap:.65rem; margin:.35rem 0 1rem; text-align:center; }
    .rca-title h1 { font-size:clamp(1.35rem, 2.2vw, 1.8rem) !important; line-height:1.25 !important; font-weight:500 !important; margin:0 !important; padding:0 !important; letter-spacing:-.015em; color:var(--rca-text); }
    .rca-title::after { content:""; width:2.5rem; height:2px; border-radius:2px; background:var(--rca-blue); opacity:.75; }

    /* panels */
    .st-key-rca-panel, .st-key-rca-columns, .st-key-rca-equation, .st-key-rca-plot, .st-key-rca-checks, .st-key-rca-colmap-inline {
        background: var(--rca-panel); border-color: var(--rca-border) !important; border-radius: 8px;
    }
    .st-key-rca-panel > [data-testid="stVerticalBlockBorderWrapper"],
    .st-key-rca-columns > [data-testid="stVerticalBlockBorderWrapper"],
    .st-key-rca-file-panel > [data-testid="stVerticalBlockBorderWrapper"],
    .st-key-rca-summary > [data-testid="stVerticalBlockBorderWrapper"],
    .st-key-rca-exports > [data-testid="stVerticalBlockBorderWrapper"],
    .st-key-rca-plot > [data-testid="stVerticalBlockBorderWrapper"] {
        padding:.85rem;
    }
    .st-key-rca-panel > [data-testid="stVerticalBlockBorderWrapper"] > [data-testid="stVerticalBlock"],
    .st-key-rca-columns > [data-testid="stVerticalBlockBorderWrapper"] > [data-testid="stVerticalBlock"],
    .st-key-rca-file-panel > [data-testid="stVerticalBlockBorderWrapper"] > [data-testid="stVerticalBlock"],
    .st-key-rca-summary > [data-testid="stVerticalBlockBorderWrapper"] > [data-testid="stVerticalBlock"],
    .st-key-rca-exports > [data-testid="stVerticalBlockBorderWrapper"] > [data-testid="stVerticalBlock"] {
        gap:.65rem;
    }
    .rca-label {
        color:var(--rca-muted); font-size:.82rem; font-weight:700;
        line-height:1.2; letter-spacing:.07em; text-transform:uppercase;
        margin:0;
    }
    .st-key-rca-file-panel [data-testid="stWidgetLabel"] { margin:0; }
    .st-key-rca-file-panel [data-testid="stWidgetLabel"] p {
        color:var(--rca-muted) !important; font-size:.82rem !important;
        font-weight:700 !important; line-height:1.2 !important;
        letter-spacing:.07em; text-transform:uppercase;
    }
    .rca-file { color:var(--rca-muted); font-size:.78rem; }
    .rca-column-status {
        color:#71869a; font-size:.72rem; line-height:1.35; margin:0;
    }
    .rca-columns-copy {
        display:flex; flex-direction:column; align-items:flex-start;
        gap:.3rem; margin:0; padding:0;
    }
    .rca-columns-copy .rca-label { font-size:.82rem; line-height:1.15; }
    .rca-columns-copy .rca-file { font-size:.76rem; line-height:1.3; }
    .rca-columns-copy .rca-column-status {
        font-size:.69rem; line-height:1.3; margin-top:.05rem;
    }

    /* file uploader: inviting without taking over the working surface */
    [data-testid="stFileUploaderDropzone"] { display:flex; flex-direction:column; align-items:center; justify-content:center; background:var(--rca-panel-2); border:1px dashed #6f8dab; border-radius:8px; padding:.7rem; transition:border-color .15s ease, background .15s ease; }
    .st-key-rca-file-panel [data-testid="stFileUploader"] {
        display:flex; flex-direction:column; gap:.65rem; height:100%;
    }
    .st-key-rca-file-panel [data-testid="stFileUploaderDropzone"] { flex:1; min-height:0; }
    .st-key-rca-upload-below [data-testid="stFileUploaderDropzone"] { min-height:8rem; padding:1.2rem; }
    [data-testid="stFileUploaderDropzone"]:hover { background:#e7f2fb; border-color:var(--rca-blue); }
    [data-testid="stFileUploaderDropzoneInstructions"] { display:none; }
    /* This workflow accepts one file. Remove it before choosing a replacement. */
    [data-testid="stFileUploaderDropzone"] button[aria-label="Add files"] { display:none; }

    /* first-run welcome card */
    .rca-welcome { max-width:55rem; margin:.8rem auto 0; padding:1.5rem 1.6rem 1.35rem; background:linear-gradient(135deg, #ffffff 0%, #f1f8fd 72%); border:1px solid var(--rca-border); border-radius:12px; box-shadow:0 14px 30px rgba(50,91,124,.10); }
    .rca-welcome-kicker { color:var(--rca-blue); font-size:.72rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; margin-bottom:.45rem; }
    .rca-welcome h2 { color:var(--rca-text); font-size:1.45rem; line-height:1.2; margin:0 0 .45rem; letter-spacing:-.02em; }
    .rca-welcome p { margin:.2rem 0; color:#314b61; font-size:.95rem; line-height:1.45; }
    .rca-welcome-steps { display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:.7rem; margin-top:1.15rem; }
    .rca-welcome-step { padding:.7rem .75rem; background:#f7fbfe; border:1px solid var(--rca-border); border-radius:8px; }
    .rca-welcome-step b { display:block; color:var(--rca-text); font-size:.85rem; margin-bottom:.2rem; text-align:center; }
    .rca-welcome-step span { display:block; color:var(--rca-muted); font-size:.78rem; line-height:1.35; text-align:center; }
    .st-key-rca-upload-below { max-width:55rem; margin:.9rem auto 0; background:var(--rca-panel); border-color:var(--rca-border) !important; border-radius:12px; }
    [role="tooltip"], [data-baseweb="tooltip"] { max-width:27rem !important; width:27rem !important; font-size:.88rem !important; line-height:1.45 !important; }
    @media (max-width: 760px) { .rca-welcome-steps { grid-template-columns:1fr; } }

    /* buttons: one shared height and rhythm for a coherent control row */
    .stButton button, .stDownloadButton button, [data-testid="stPopoverButton"] {
        height: auto; min-height: 2.35rem; padding: .35rem .8rem; border-radius: 7px; font-size: .85rem; font-weight: 600; line-height:1.2;
        display:inline-flex; align-items:center; justify-content:center;
    }
    [data-testid="stBaseButton-primary"] { background:#2478e5; border:1px solid #3d8bf0; color:#fff; }
    [data-testid="stBaseButton-primary"],
    [data-testid="stBaseButton-primary"] p,
    [data-testid="stBaseButton-primary"] span,
    [data-testid="stBaseButton-primary"] [data-testid="stIconMaterial"] { color:#fff !important; }
    [data-testid="stBaseButton-primary"]:hover { background:#3188f4; border-color:#7eb9ff; color:#fff; }
    [data-testid="stBaseButton-secondary"], [data-testid="stPopoverButton"] { background:var(--rca-panel-2); border:1px solid #9bbbd3; color:var(--rca-text); }
    [data-testid="stBaseButton-secondary"]:hover, [data-testid="stPopoverButton"]:hover { border-color:var(--rca-blue); color:var(--rca-blue); }
    .stButton button p, .stDownloadButton button p, [data-testid="stPopoverButton"] p { font-size: inherit; }
    .stDownloadButton, .stButton, [data-testid="stPopover"] { align-self:stretch; }
    .stDownloadButton button, .stButton button, [data-testid="stPopoverButton"] { width:100%; }
    button:focus-visible, [role="tab"]:focus-visible, input:focus-visible { outline: 2px solid var(--rca-focus) !important; outline-offset: 2px; }

    /* status line + alerts: one line, not a slab */
    [data-testid="stAlertContainer"] { display:flex; align-items:center; min-height:2.8rem; box-sizing:border-box; padding: .35rem .8rem; border-radius: 6px; }
    [data-testid="stAlertContainer"] p { font-size: .9rem; margin:0; }
    [data-testid="stAlertContainer"] [data-testid^="stAlertContent"] > div { align-items:center; }
    [data-testid="stAlertContainer"] div:has(> [data-testid="stAlertDynamicIcon"]) { top:0; display:flex; align-items:center; }
    [data-testid="stAlertContainer"] [data-testid="stMarkdownContainer"] { margin:0; }
    [data-testid="stAlertContainer"] p { line-height:1.4; }

    /* Center control text and glyphs without constraining tooltip wrappers. */
    button [data-testid="stMarkdownContainer"],
    [data-testid="stCheckbox"] [data-testid="stMarkdownContainer"] { margin:0; }
    button [data-testid="stMarkdownContainer"] p { margin:0; line-height:1.25; }
    button [data-testid="stIconMaterial"],
    [data-testid="stAlertContainer"] [data-testid="stIconMaterial"] {
        display:inline-flex; align-items:center; justify-content:center;
        width:1.1em; height:1.1em; line-height:1; flex-shrink:0;
    }
    [data-testid="stCheckbox"] label { align-items:center; }
    [data-testid="stCheckbox"] [data-testid="stWidgetLabel"] { align-items:center; }
    [data-testid="stCheckbox"] p { margin:0; line-height:1.4; }
    button[aria-label^="Help for "] {
        display:inline-flex; align-items:center; justify-content:center;
        width:1.25rem; height:1.25rem; min-height:0; padding:0; flex-shrink:0;
    }
    button[aria-label^="Help for "] svg { width:1rem; height:1rem; }

    /* The curve is the primary work surface; the rail holds a compact table. */
    .st-key-rca-summary, .st-key-rca-file-panel, .st-key-rca-exports {
        background:var(--rca-panel); border-color:var(--rca-border) !important; border-radius:8px;
    }
    .rca-chart-heading { text-align:center; padding:.5rem .25rem .8rem; margin-bottom:0; }
    .st-key-rca-fit-status [data-testid="stAlertContainer"] { justify-content:center; text-align:center; }
    .st-key-rca-fit-status [data-testid="stAlertContainer"] {
        min-height:3.15rem; padding:.4rem .85rem; border-radius:8px;
    }
    .st-key-rca-fit-status [data-testid="stAlertContainer"] p { font-size:.95rem; }
    .st-key-rca-fit-status [data-testid="stAlertDynamicIcon"] { font-size:1.1rem; }
    .st-key-rca-fit-status [data-testid^="stAlertContent"] {
        flex:0 1 auto !important; width:auto !important;
    }
    .st-key-rca-fit-status [data-testid^="stAlertContent"] > div { justify-content:center; }
    .st-key-rca-fit-status [data-testid="stAlertContainer"] [data-testid="stMarkdownContainer"] {
        flex:0 1 auto; width:auto !important; text-align:center;
    }
    .rca-chart-heading h2 { font-size:1.25rem; font-weight:500; margin:0 0 .5rem; padding:0; }
    .rca-chart-heading code { background:transparent; color:var(--rca-muted); font-size:1.15rem; white-space:normal; overflow-wrap:anywhere; }
    .st-key-rca-equation .rca-chart-heading {
        min-height:3rem; width:100%; box-sizing:border-box; padding:.3rem 1rem;
        display:flex; align-items:center; justify-content:center;
        transform:none;
    }
    .st-key-rca-equation [data-testid="stMarkdownContainer"] {
        width:100%; margin:0 !important;
    }
    .st-key-rca-plot [data-testid="stImage"] { position:relative; }
    .st-key-rca-plot [data-testid="stElementToolbar"] {
        top:.75rem !important; right:.75rem !important; padding:0 !important;
        transform:none !important;
    }
    .st-key-rca-plot button[aria-label="Fullscreen"] {
        position:static !important; inset:auto !important; transform:none !important;
        width:1.8rem !important; height:1.8rem !important; min-height:1.8rem !important;
        padding:.3rem !important; border:1px solid #c6d6e2 !important;
        border-radius:6px !important; background:rgba(255,255,255,.94) !important;
        color:#24415b !important; box-shadow:0 2px 8px rgba(31,65,91,.14);
        z-index:5;
    }
    .st-key-rca-plot button[aria-label="Fullscreen"] svg {
        width:.9rem !important; height:.9rem !important;
    }
    .rca-summary-table {
        overflow:hidden; border:1px solid #d8dee3; border-radius:8px;
        background:#fff; margin-bottom:1rem;
    }
    .rca-summary-row {
        display:grid; grid-template-columns:minmax(0,47%) minmax(0,53%);
        align-items:center; min-height:2.2rem; border-top:1px solid #e1e5e8;
    }
    .rca-summary-row:first-child { border-top:0; }
    .rca-summary-cell {
        min-width:0; padding:.4rem .5rem; color:var(--rca-text);
        font-size:.76rem; line-height:1.2;
    }
    .rca-summary-cell + .rca-summary-cell { border-left:1px solid #e1e5e8; }
    .rca-summary-row:not(:first-child) .rca-summary-cell:first-child { font-weight:400; }
    .rca-summary-row:first-child {
        background:#fff; border-bottom:2px solid #c4ccd2;
    }
    .rca-summary-head { color:var(--rca-text); font-size:.76rem; font-weight:400; }
    .rca-summary-value { text-align:right; font-variant-numeric:tabular-nums; }
    .rca-summary-number { display:block; white-space:nowrap; font-size:.82rem; font-weight:400; }
    .rca-summary-note {
        display:block; margin-top:.16rem; color:var(--rca-muted);
        font-size:.65rem; font-weight:400; line-height:1.25;
    }
    .st-key-rca-top-row [data-testid="stColumn"],
    .st-key-rca-workspace [data-testid="stColumn"] { min-width:0; }
    .st-key-rca-top-row > [data-testid="stHorizontalBlock"] { align-items:stretch; }
    .st-key-rca-top-row [data-testid="stColumn"] > [data-testid="stVerticalBlock"] { height:100%; gap:.5rem; }
    .st-key-rca-columns, .st-key-rca-file-panel {
        height:100%; min-height:8.7rem; box-sizing:border-box;
    }
    .st-key-rca-columns > [data-testid="stVerticalBlockBorderWrapper"],
    .st-key-rca-file-panel > [data-testid="stVerticalBlockBorderWrapper"] { height:100%; }
    .st-key-rca-columns > [data-testid="stVerticalBlockBorderWrapper"] > [data-testid="stVerticalBlock"] {
        height:100%; justify-content:flex-start; gap:.5rem;
    }
    .st-key-rca-columns [data-testid="stPopover"] { margin-top:auto; }
    .st-key-rca-columns [data-testid="stPopoverButton"] {
        min-height:2.35rem; font-size:.84rem; line-height:1.2;
    }
    .st-key-rca-file-panel > [data-testid="stVerticalBlockBorderWrapper"] > [data-testid="stVerticalBlock"] {
        height:100%;
    }
    .st-key-rca-workspace [data-testid="stColumn"] > [data-testid="stVerticalBlock"] { gap:.5rem; }
    @media (min-width:1101px) and (min-height:700px) {
        .st-key-rca-columns { position:relative; }
        .st-key-rca-columns [data-testid="stPopover"] {
            position:absolute; left:.85rem; right:.85rem; bottom:.85rem;
            width:auto !important; margin:0;
        }
        .st-key-rca-plot img { width:100%; max-height:calc(100svh - 18rem); object-fit:contain; }
        .rca-summary-cell { padding:.3rem .42rem; }
        .rca-chart-heading { padding:.25rem .25rem .5rem; }
        .rca-title { margin:.1rem 0 .65rem; gap:.4rem; }
        .rca-screen-nav { padding:0; margin:0; }
    }
    @media (min-width:761px) and (max-width:1100px) {
        .st-key-rca-top-row > [data-testid="stHorizontalBlock"],
        .st-key-rca-workspace > [data-testid="stHorizontalBlock"] { flex-wrap:wrap; }
        .st-key-rca-top-row > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
        .st-key-rca-workspace > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { min-width:14rem; }
    }

    /* quality checks: icon + title + detail, never colour alone */
    .rca-chk { display:grid; grid-template-columns: 1.25rem 1fr; gap:.45rem; padding:.42rem 0; border-top:1px solid var(--rca-border); font-size:.8rem; line-height:1.35; color:#314b61; }
    .rca-chk:first-of-type { border-top:0; }
    .rca-chk b { display:block; color:var(--rca-text); font-weight:600; font-size:.84rem; }
    .rca-chk-ic { width:1.25rem; height:1.25rem; border-radius:50%; display:grid; place-items:center; line-height:1; font-size:.72rem; font-weight:600; color:var(--rca-text); border:2px solid currentColor; box-sizing:border-box; }
    .rca-chk-ic svg { width:70%; height:70%; display:block; }
    .rca-ok .rca-chk-ic { color:var(--rca-green); }
    .rca-warn .rca-chk-ic { color:var(--rca-amber); }
    .rca-bad .rca-chk-ic { color:var(--rca-red); }
    .rca-info .rca-chk-ic { color:var(--rca-blue); }
    .rca-empty { max-width: 40rem; margin-top: .4rem; }

    /* Two full-height result screens share the page's native scroll area. */
    [data-testid="stMain"]:has(.st-key-rca-details) {
        scroll-snap-type:y mandatory;
        scroll-padding-top:1rem;
    }
    .st-key-rca-overview, .st-key-rca-details {
        min-height:calc(100svh - 2.5rem);
        scroll-snap-align:start;
        scroll-snap-stop:always;
    }
    .st-key-rca-details {
        margin-top:2rem; padding:1.5rem;
        background:#e1f0fa; border:1px solid #afcce0; color:var(--rca-text);
        border-radius:12px;
    }
    .st-key-rca-details p, .st-key-rca-details label,
    .st-key-rca-details [data-testid="stMarkdownContainer"] { color:#314b61; }
    .st-key-rca-details [data-testid="stDataFrame"],
    .st-key-rca-details [data-testid="stTable"] {
        background:#ffffff; border-radius:8px; box-shadow:0 1px 3px rgba(37,78,108,.08);
    }
    .st-key-rca-details [data-testid="stDataFrame"] [role="columnheader"],
    .st-key-rca-details [data-testid="stDataFrame"] [role="row"]:first-child [role="cell"],
    .st-key-rca-details [data-testid="stTable"] th {
        color:#000 !important; fill:#000 !important;
        border-bottom:2px solid #8999a6 !important;
    }
    .st-key-rca-details [data-testid="stDataFrame"] [role="columnheader"] * {
        color:#000 !important; fill:#000 !important;
    }
    .st-key-rca-rating-table { position:relative; }
    .st-key-rca-rating-table [data-testid="stDataFrame"] { position:relative; }
    .st-key-rca-rating-table [data-testid="stDataFrame"]::after {
        content:""; position:absolute; z-index:4; pointer-events:none;
        left:0; right:0; top:2.2rem; height:2px; background:#8999a6;
    }
    .st-key-rca-rating-table [data-testid="stMarkdownContainer"]:has(> .rca-grid-header) {
        position:relative; z-index:20;
        height:0; min-height:0; margin:0 !important; overflow:visible;
        transform:translateY(calc(-660px - .45rem));
        pointer-events:none;
    }
    .rca-grid-header {
        position:absolute; inset:0; height:2.2rem;
        display:grid;
        grid-template-columns:10.1% 13.1% 15.55% 15.6% 15.15% 15.35% 15.15%;
        overflow:hidden; box-sizing:border-box; pointer-events:none;
        background:#fff; border-radius:8px 8px 0 0;
        border:1px solid var(--rca-border); border-bottom:0;
    }
    .rca-grid-header > span {
        min-width:0; padding:.4rem .55rem; display:flex; align-items:center;
        color:#000; font-size:.82rem; font-weight:400; line-height:1.2;
        white-space:nowrap; overflow:hidden; text-overflow:clip;
        border-left:1px solid #d8dee3;
    }
    .rca-grid-header > span:first-child {
        border-left:0; justify-content:space-between;
    }
    .rca-grid-header-menu {
        color:#000; font-size:1rem; line-height:1; margin-left:.35rem;
    }
    .st-key-rca-overview > [data-testid="stVerticalBlock"] { min-height:calc(100svh - 2.5rem); }
    .st-key-rca-overview [data-testid="stMarkdownContainer"]:has(> .rca-screen-nav) {
        flex:1 1 auto; min-height:clamp(3rem, 6svh, 4.5rem);
        display:flex; align-items:center; justify-content:center;
    }
    .rca-screen-nav {
        display:flex; align-items:center; justify-content:center;
        width:100%; height:100%; padding:0; margin:0; position:relative; z-index:2;
    }
    .rca-screen-nav a, .rca-details-heading a {
        color:var(--rca-blue); text-decoration:none; font-size:.85rem;
    }
    .rca-screen-nav a:hover, .rca-details-heading a:hover { text-decoration:underline; }
    .rca-details-heading { display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap; margin-bottom:.75rem; }
    .rca-details-heading h2 { margin:0; padding:0; font-size:1.5rem; font-weight:500; }
    #rca-overview, #rca-details { scroll-margin-top:1.5rem; }
    @media (prefers-reduced-motion:no-preference) {
        [data-testid="stMain"]:has(.st-key-rca-details) { scroll-behavior:smooth; }
    }
    @media (max-width:760px) {
        [data-testid="stMain"]:has(.st-key-rca-details) { scroll-snap-type:y proximity; }
        .st-key-rca-details { padding:1rem; }
    }

    /* tabs + expanders */
    [data-testid="stTabs"] [role="tab"] p { font-size:.86rem; }
    .st-key-rca-details [role="tab"][aria-selected="true"] p { color:var(--rca-blue) !important; }
    .st-key-rca-details [data-baseweb="tab-highlight"] { background-color:var(--rca-blue) !important; }
    [data-testid="stExpander"] details { border-color:var(--rca-border); background:#ffffff; border-radius:6px; }
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


_CHECK_PATH = {
    "ok": '<path d="m3 8 3 3 7-7"/>',
    "warn": '<path d="M8 3v6m0 3v.1"/>',
    "bad": '<path d="m4 4 8 8m0-8-8 8"/>',
    "info": '<path d="M8 7v6m0-10v.1"/>',
}
_CHECK_ICON = {
    level: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" '
           'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
           + path + '</svg>'
    for level, path in _CHECK_PATH.items()
}


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
# Layout: title, result heading, then controls | graph | summary.
# --------------------------------------------------------------------------- #
overview = st.container(key="rca-overview")
overview.markdown(
    '<div class="rca-title" id="rca-overview"><h1>Rating Curve Automater</h1></div>',
    unsafe_allow_html=True,
)

def _measurement_uploader():
    return st.file_uploader(
        "Measurement spreadsheet",
        type=["xlsx", "xls", "csv"],
        key="measurement_upload",
        accept_multiple_files=False,
        help="Accepted: XLSX, XLS, or CSV up to 200MB.  \n"
             "We find date, stage, and discharge columns, then clean invalid rows and detect units before fitting.  \n"
             "One file at a time. Remove the current file with × to choose another.",
    )


# Keep the first-run action directly below the welcome card. Once a file is
# loaded, move the same uploader into the right-hand file panel.
show_side_upload = bool(st.session_state.get("measurement_upload"))
if show_side_upload:
    with overview.container(key="rca-top-row"):
        top_left, top_center, top_right = st.columns([1.05, 3.35, 1.25], gap="small")
    column_controls = top_left.container(border=True, key="rca-columns")
    status_box = top_center.container(key="rca-fit-status")
    equation_box = top_center.container(border=True, key="rca-equation")
    file_box = top_right.container(border=True, key="rca-file-panel")
    with overview.container(key="rca-workspace"):
        side, center, rail = st.columns([1.05, 3.35, 1.25], gap="small")
    panel = side.container(border=True, key="rca-panel")
    fit_controls = panel.container()
    main = center.container(border=True, key="rca-plot")
    summary_box = rail.container(border=True, key="rca-summary")
    export_box = rail.container(border=True, key="rca-exports")
else:
    side = None
    main = overview.container()

if show_side_upload:
    with file_box:
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

# Move a newly selected file into the working layout immediately.
if not show_side_upload:
    st.rerun()

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
with column_controls:
    mapping_status = (
        "Required fields found."
        if pre.mapping.is_complete
        else "Required fields not found."
    )
    st.markdown(
        '<div class="rca-columns-copy">'
        '<div class="rca-label">Data columns</div>'
        f'<div class="rca-file">{pre.n_rows:,} rows · {len(pre.source_columns)} columns</div>'
        f'<div class="rca-column-status">{mapping_status}</div>'
        '</div>',
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
with fit_controls:
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

    with st.container(gap="small"):
        with st.popover("Uncertainty", icon=":material/tune:", width="stretch",
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
        with st.popover("Advanced", icon=":material/more_horiz:", width="stretch",
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
with equation_box:
    st.markdown(
        '<div class="rca-chart-heading">'
        f'<code>{html.escape(_display_equation(p))}</code></div>',
        unsafe_allow_html=True,
    )
with status_box:
    if not outcome.is_plausible:
        st.error("**Not a plausible rating curve** — see the Checks tab in Detailed results.",
                 icon=":material/error:")
    elif outcome.warnings:
        st.warning(f"**Fitted, with warnings** · {r2_label} = {r2_value:.3f} "
                   f"from {p['n_points']} gaugings", icon=":material/warning:")
    else:
        st.success(f"**Rating curve fitted** · {r2_label} = {r2_value:.3f} "
                   f"from {p['n_points']} gaugings", icon=":material/check_circle:")

with export_box:
    st.markdown('<div class="rca-label">Export</div>', unsafe_allow_html=True)
    st.download_button(
        "Excel report", data=report_bytes, type="primary", icon=":material/download:", width="stretch",
        file_name=f"rating_curve_report{tag}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Data, fitted curve, uncertainty, diagnostics and the rating table in one workbook.",
    )
    st.download_button(
        "Rating table", data=rating_csv, icon=":material/table:", width="stretch",
        file_name=f"rating_table{tag}.csv", mime="text/csv",
        help=f"Stage → discharge lookup every {rating_step:g} m, as CSV.",
    )


# --------------------------------------------------------------------------- #
# Headline numbers
# --------------------------------------------------------------------------- #
with summary_box:
    valid_stage = cleaned.loc[cleaned["is_valid"], STAGE_M]
    stage_min = float(valid_stage.min()) if not valid_stage.empty else float("nan")
    stage_max = float(valid_stage.max()) if not valid_stage.empty else float("nan")
    quiet = dict(delta_color="off", delta_arrow="off")

    st.markdown('<div class="rca-label">Fit summary</div>', unsafe_allow_html=True)
    summary_rows = []

    def summary_row(label, value, delta=None, help=None, **_):
        summary_rows.append({"Parameter": str(label), "Value": str(value), "Note": str(delta or "")})
    if p.get("is_segmented"):
        breaks = ", ".join(f"{x:.3f}" for x in p.get("breakpoints") or [])
        summary_row("Segments", p["n_segments"], delta=f"breaks at {breaks} m" if breaks else None,
                       help="Per-segment a and b are listed under Fit details.", **quiet)
    else:
        summary_row("a", f"{p['a']:.4f}", help="Coefficient in Q = a·(H − h₀)^b.")
        if p.get("b_fixed"):
            b_note = "imposed"
        elif bands and bands.get("b_ci"):
            b_note = f"{pct}% CI {bands['b_ci'][0]:.2f}–{bands['b_ci'][1]:.2f}"
        else:
            b_note = None
        summary_row("b", f"{p['b']:.3f}", delta=b_note, help="Exponent in Q = a·(H − h₀)^b.", **quiet)
    if not p["h0_estimated"]:
        h0_note, h0_help = "set by hand", "Stage of zero flow, set by hand."
    else:
        h0_note = "weakly identified" if hd.get("railed") else "estimated"
        h0_help = f"Stage of zero flow, estimated from the low-flow gaugings ({hd.get('method', '?')} method)."
    summary_row("h₀ (m)", f"{p['h0']:.3f}", delta=h0_note,
                   delta_color="orange" if hd.get("railed") else "off", delta_arrow="off",
                   help=h0_help)
    summary_row(r2_label, f"{r2_value:.3f}")
    summary_row("Valid rows", result.valid_count,
                   delta=f"{result.invalid_count} excluded" if result.invalid_count else None,
                   delta_color="orange" if result.invalid_count else "off", delta_arrow="off")
    summary_row("Warnings (kept)", result.warning_count,
                   help="Rows kept in the fit but flagged — see the Row warnings tab.")
    if bands:
        unit = "draws" if bands.get("kind") == "posterior" else "refits"
        summary_row("Band at mid-stage", f"±{bands['ci_halfwidth_pct_at_median']:.0f}%",
                       delta=f"{pct}% CI · {bands['n_success']} {unit}", **quiet)
    else:
        summary_row("Band at mid-stage", "—", delta="needs ≥ 4 gaugings", **quiet)
    summary_row("Stage range (m)", f"{stage_min:.2f}–{stage_max:.2f}")
    summary_html = [
        '<div class="rca-summary-table" role="table" aria-label="Fit summary">',
        '<div class="rca-summary-row" role="row">'
        '<div class="rca-summary-cell rca-summary-head" role="columnheader">Parameter</div>'
        '<div class="rca-summary-cell rca-summary-head" role="columnheader">Value</div></div>',
    ]
    for row in summary_rows:
        label_text = html.escape(row["Parameter"], quote=True)
        value_text = html.escape(row["Value"], quote=True)
        note_text = html.escape(row["Note"], quote=True)
        note_html = f'<span class="rca-summary-note">{note_text}</span>' if note_text else ""
        summary_html.append(
            f'<div class="rca-summary-row" role="row" data-summary-label="{label_text}" '
            f'data-summary-value="{value_text}" data-summary-note="{note_text}">'
            f'<div class="rca-summary-cell" role="cell">{label_text}</div>'
            f'<div class="rca-summary-cell rca-summary-value" role="cell">'
            f'<span class="rca-summary-number">{value_text}</span>{note_html}</div></div>'
        )
    summary_html.append("</div>")
    st.markdown("".join(summary_html), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Main curve and quality checks
# --------------------------------------------------------------------------- #
with main:
    curve_figure = make_rating_curve_figure(
        fit_df, a=p["a"], b=p["b"], h0=p["h0"], log_scale=log_scale,
        fit=p, figure=Figure(figsize=(9, 6.3), dpi=160),
    )
    curve_figure.axes[0].set_title("", loc="left")  # The panel header carries the title and equation.
    curve_figure.tight_layout(pad=1.1)
    st.pyplot(curve_figure, width="stretch")

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
                             "Drawn as outlined squares; details in the Row warnings tab."))

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



# --------------------------------------------------------------------------- #
# Detail tabs — a separate full-width screen below the dashboard.
# --------------------------------------------------------------------------- #
overview.markdown(
    '<div class="rca-screen-nav"><a href="#rca-details" target="_self">Explore detailed results ↓</a></div>',
    unsafe_allow_html=True,
)
with st.container(key="rca-details"):
    st.markdown(
        '<div class="rca-details-heading" id="rca-details">'
        '<h2>Detailed results</h2></div>',
        unsafe_allow_html=True,
    )
    tab_names = ["Rating table", "Residuals over time", "Fit details", "Checks"]
    if result.invalid_count:
        tab_names.append(f"Excluded rows ({result.invalid_count})")
    if result.warning_count:
        tab_names.append(f"Row warnings ({result.warning_count})")
    tabs = dict(zip(tab_names, st.tabs(
        tab_names, default="Residuals over time" if drift and drift["flag"] == "likely" else None,
    )))

    with tabs["Checks"]:
        st.markdown('<div class="rca-label">Checks</div>' + "".join(checks), unsafe_allow_html=True)

    with tabs["Rating table"]:
        st.caption(f"Stage → discharge every {rating_step:g} m · {len(rating_table)} rows. "
                   "Change the step under Advanced.")
        with st.container(key="rca-rating-table"):
            st.dataframe(rating_table, width="stretch", height=660, hide_index=True)
            st.markdown(
                '<div class="rca-grid-header">'
                '<span>Stage (m)<b class="rca-grid-header-menu">⋮</b></span>'
                '<span>Discharge (m³/s)</span>'
                '<span>95% confidence lower</span>'
                '<span>95% confidence upper</span>'
                '<span>95% prediction lower</span>'
                '<span>95% prediction upper</span>'
                '<span>Within gauged range</span>'
                '</div>',
                unsafe_allow_html=True,
            )

    with tabs["Residuals over time"]:
        resid_fig = make_residual_time_figure(fit_df, p, figure=Figure(figsize=(11, 4.2), dpi=240))
        if resid_fig is not None:
            residual_svg = StringIO()
            resid_fig.savefig(residual_svg, format="svg", bbox_inches="tight")
            st.image(residual_svg.getvalue(), width="stretch")
        else:
            st.caption("The gaugings carry no usable dates, so residuals can't be plotted over time.")

    with tabs["Fit details"]:
        st.caption("a is the curve coefficient; b is the exponent; h₀ is the stage of zero flow. "
                   "Intervals in the fit summary describe parameter uncertainty.")
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
                         width="stretch", height=560, hide_index=True)
    if result.warning_count:
        with tabs[f"Row warnings ({result.warning_count})"]:
            cols = [c for c in (DATE, STAGE_M, DISCHARGE_CMS, "warning_notes") if c in cleaned.columns]
            st.dataframe(_friendly(cleaned.loc[cleaned["has_warning"], cols]),
                         width="stretch", height=560, hide_index=True)
