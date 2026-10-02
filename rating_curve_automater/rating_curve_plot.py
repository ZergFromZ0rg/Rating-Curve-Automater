"""Matplotlib rating-curve figure, shared by the GUI preview."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rating_curve_automater.rating_curve_fitting import predict_discharge, select_valid_measurements
from rating_curve_automater.schema import DISCHARGE_CMS, STAGE_M

OBSERVED_COLOR = "#111111"
MODEL_COLOR = "#111111"
WARNING_COLOR = "#111111"


def _style_axes(ax):
    """Restrained, consistent drafting style for the on-screen plots."""
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", colors="#333333",
                   direction="out", length=3, width=0.6, labelsize=9)
    ax.tick_params(axis="both", which="minor", length=2, width=0.5)
    for spine in ax.spines.values():
        spine.set_color("#444444")
        spine.set_linewidth(0.6)
    ax.grid(False, which="both")
    ax.grid(True, which="major", color="#e3e5e7", linewidth=0.45)
    for label in (ax.xaxis.label, ax.yaxis.label):
        label.set_fontsize(10)
        label.set_fontfamily("sans-serif")
    ax.xaxis.labelpad = 8
    ax.yaxis.labelpad = 8


def _legend(ax):
    ax.legend(fontsize=8, facecolor="white", edgecolor="none",
              framealpha=0.95, labelcolor="#333333", handlelength=2.4,
              borderpad=0.7, labelspacing=0.55, scatterpoints=1)


def make_rating_curve_figure(
    df: pd.DataFrame,
    a: float,
    b: float,
    h0: float,
    figure=None,
    log_scale: bool = False,
    fit: dict | None = None,
):
    """Draw observed points and the fitted curve onto a Matplotlib figure.

    A ``figure`` may be supplied (e.g. one already bound to a Tk canvas); it is
    cleared and reused. Otherwise a new one is created. Pass ``fit`` (the dict
    from :func:`fit_rating_curve`) to render a segmented curve.
    """
    from matplotlib.figure import Figure

    working = select_valid_measurements(df)
    stage = working[STAGE_M].to_numpy(dtype=float)
    observed = working[DISCHARGE_CMS].to_numpy(dtype=float)

    fig = figure if figure is not None else Figure(figsize=(8.8, 3.65), dpi=130)
    fig.clear()
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    warned = None
    if "has_warning" in working.columns:
        warned = working["has_warning"].to_numpy(dtype=bool)

    if warned is not None and warned.any():
        ax.scatter(stage[~warned], observed[~warned], s=13, facecolors="white", edgecolors=OBSERVED_COLOR, linewidths=0.65, label="Observed", zorder=3)
        ax.scatter(stage[warned], observed[warned], s=17, facecolors="white", edgecolors=WARNING_COLOR, linewidths=0.7, marker="s", label="Observed (warning)", zorder=4)
    else:
        ax.scatter(stage, observed, s=13, facecolors="white", edgecolors=OBSERVED_COLOR, linewidths=0.65, label="Observed", zorder=3)

    curve_hi = float(stage.max())
    _manning = fit.get("manning") if fit is not None else None
    if _manning and _manning.get("extrapolation_ceiling"):
        curve_hi = max(curve_hi, float(_manning["extrapolation_ceiling"]))
    curve_stage = np.linspace(float(stage.min()), curve_hi, 1000)
    curve_stage = curve_stage[curve_stage > h0]
    if fit is not None and fit.get("is_segmented"):
        modeled = predict_discharge(fit, curve_stage)
        bps = fit.get("breakpoints", [fit.get("breakpoint")])
        label = f"{fit.get('n_segments', len(bps) + 1)} segments (breaks H={', '.join(f'{b:.3f}' for b in bps)})"
    else:
        modeled = a * np.power(curve_stage - h0, b)
        label = f"Q = {a:.3f}·(H−{h0:.3f})^{b:.3f}"
    bands = fit.get("bands") if fit is not None else None
    if bands:
        gs = np.asarray(bands["stage"], dtype=float)
        pct = int(round(bands["level"] * 100))
        ax.fill_between(
            gs, bands["pi_lower"], bands["pi_upper"],
            color=MODEL_COLOR, alpha=0.055, linewidth=0,
            label=f"{pct}% prediction", zorder=1,
        )
        ax.fill_between(
            gs, bands["ci_lower"], bands["ci_upper"],
            color=MODEL_COLOR, alpha=0.12, linewidth=0,
            label=f"{pct}% confidence", zorder=1,
        )

    ax.plot(curve_stage, modeled, color=MODEL_COLOR, linewidth=1.05, solid_capstyle="round", solid_joinstyle="round", antialiased=True, label=label, zorder=2)
    if fit is not None and fit.get("is_segmented"):
        bp_ci = fit.get("breakpoint_ci") or []
        bp_pct = int(round((fit.get("bands") or {}).get("level", 0.95) * 100))
        for i, bp in enumerate(fit.get("breakpoints", [fit.get("breakpoint")])):
            if i < len(bp_ci):
                lo, hi = bp_ci[i]
                ax.axvspan(lo, hi, color="#7f7f7f", alpha=0.12, linewidth=0, zorder=0,
                           label=f"breakpoint {bp_pct}% CI" if i == 0 else None)
            ax.axvline(bp, color="#7f7f7f", linestyle=(0, (4, 4)), linewidth=0.65, zorder=1)

    manning = fit.get("manning") if fit is not None else None
    if manning and "stage" in manning:
        ms = np.asarray(manning["stage"], dtype=float)
        ax.plot(ms, np.asarray(manning["q_manning"], dtype=float),
                color="#333333", linewidth=0.95, linestyle=(0, (5, 3)), zorder=2,
                label=f"Manning (n={manning.get('n_used', float('nan')):.3f}) [{manning['flag']}]")
        ax.axvline(manning["stage_max_gauged"], color="#333333", linestyle=":",
                   linewidth=1, alpha=0.7, zorder=1)

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel("Stage (m)", color="#111111")
    ax.set_ylabel("Discharge (m³/s)", color="#111111")
    ax.set_title("Fitted rating curve", color="#111111", loc="left", pad=12, fontweight="bold")
    _style_axes(ax)
    _legend(ax)
    fig.tight_layout()
    return fig


def make_residual_time_figure(df: pd.DataFrame, fit: dict, figure=None):
    """Residual (%) of each gauging against its date, or ``None`` when the
    gaugings carry no usable dates. A fitted time trend is drawn when
    ``fit['drift']`` reports one."""
    from matplotlib.figure import Figure

    from rating_curve_automater.rating_curve_drift import OUT_LOG, OUT_PCT, build_residual_frame

    working = select_valid_measurements(df)
    frame = build_residual_frame(working, fit)
    if frame is None:
        return None

    dates = pd.to_datetime(frame.attrs["dates"]).reset_index(drop=True)
    resid = frame[OUT_PCT].to_numpy(dtype=float)
    log_resid = frame[OUT_LOG].to_numpy(dtype=float)

    fig = figure if figure is not None else Figure(figsize=(10.0, 4.2), dpi=220)
    fig.clear()
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    order = np.argsort(dates.to_numpy())
    dates = dates.iloc[order].reset_index(drop=True)
    resid = resid[order]
    log_resid = log_resid[order]

    ax.axhline(0.0, color="#66727d", linewidth=0.75, zorder=1)
    ax.scatter(
        dates, resid, s=18, facecolors="white", edgecolors=OBSERVED_COLOR,
        linewidths=0.8, zorder=3, label="Gauging residual",
    )

    drift = fit.get("drift") or {}
    rate = drift.get("trend_pct_per_year")
    if rate is not None:
        t_years = (dates - dates.iloc[0]).dt.total_seconds().to_numpy() / (365.25 * 86400.0)
        slope = float(np.log1p(rate / 100.0))
        intercept = float(log_resid.mean() - slope * t_years.mean())
        trend_pct = (np.exp(intercept + slope * t_years) - 1.0) * 100.0
        ax.plot(dates, trend_pct, color=MODEL_COLOR, linewidth=1.15,
                label=f"trend {rate:+.1f}%/yr", zorder=2)
        _legend(ax)

    ax.set_xlabel("Gauging date", color="#111111")
    ax.set_ylabel("Observed − modelled (%)", color="#111111")
    ax.set_title("Rating-curve residuals over time", color="#111111", loc="left", fontweight="bold")
    _style_axes(ax)
    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
    locator = AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
    ax.margins(x=0.025, y=0.12)
    fig.tight_layout(pad=1.4)
    return fig
