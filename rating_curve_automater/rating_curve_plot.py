"""Matplotlib rating-curve figure, shared by the GUI preview."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rating_curve_automater.rating_curve_fitting import predict_discharge, select_valid_measurements
from rating_curve_automater.schema import DISCHARGE_CMS, STAGE_M

OBSERVED_COLOR = "#1f77b4"
MODEL_COLOR = "#d62728"
WARNING_COLOR = "#ff7f0e"


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

    fig = figure if figure is not None else Figure(figsize=(6.4, 3.8))
    fig.clear()
    ax = fig.add_subplot(111)

    warned = None
    if "has_warning" in working.columns:
        warned = working["has_warning"].to_numpy(dtype=bool)

    if warned is not None and warned.any():
        ax.scatter(stage[~warned], observed[~warned], s=20, color=OBSERVED_COLOR, label="Observed", zorder=3)
        ax.scatter(stage[warned], observed[warned], s=32, color=WARNING_COLOR, marker="s", label="Observed (warning)", zorder=4)
    else:
        ax.scatter(stage, observed, s=20, color=OBSERVED_COLOR, label="Observed", zorder=3)

    curve_hi = float(stage.max())
    _manning = fit.get("manning") if fit is not None else None
    if _manning and _manning.get("extrapolation_ceiling"):
        curve_hi = max(curve_hi, float(_manning["extrapolation_ceiling"]))
    curve_stage = np.linspace(float(stage.min()), curve_hi, 300)
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
            color=MODEL_COLOR, alpha=0.10, linewidth=0,
            label=f"{pct}% prediction", zorder=1,
        )
        ax.fill_between(
            gs, bands["ci_lower"], bands["ci_upper"],
            color=MODEL_COLOR, alpha=0.25, linewidth=0,
            label=f"{pct}% confidence", zorder=1,
        )

    ax.plot(curve_stage, modeled, color=MODEL_COLOR, linewidth=2, label=label, zorder=2)
    if fit is not None and fit.get("is_segmented"):
        for bp in fit.get("breakpoints", [fit.get("breakpoint")]):
            ax.axvline(bp, color="#7f7f7f", linestyle="--", linewidth=1, zorder=1)

    manning = fit.get("manning") if fit is not None else None
    if manning and "stage" in manning:
        ms = np.asarray(manning["stage"], dtype=float)
        ax.plot(ms, np.asarray(manning["q_manning"], dtype=float),
                color="#8c564b", linewidth=1.6, linestyle=(0, (5, 2)), zorder=2,
                label=f"Manning (n={manning.get('n_used', float('nan')):.3f}) [{manning['flag']}]")
        ax.axvline(manning["stage_max_gauged"], color="#8c564b", linestyle=":",
                   linewidth=1, alpha=0.7, zorder=1)

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel("Stage above bed (m)")
    ax.set_ylabel("Discharge (m³/s)")
    ax.set_title("Rating curve")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
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

    fig = figure if figure is not None else Figure(figsize=(6.4, 3.0))
    fig.clear()
    ax = fig.add_subplot(111)

    ax.axhline(0.0, color="#7f7f7f", linewidth=1, zorder=1)
    ax.scatter(dates, resid, s=22, color=OBSERVED_COLOR, zorder=3)

    drift = fit.get("drift") or {}
    rate = drift.get("trend_pct_per_year")
    if rate is not None:
        t_years = (dates - dates.iloc[0]).dt.total_seconds().to_numpy() / (365.25 * 86400.0)
        slope = float(np.log1p(rate / 100.0))
        intercept = float(log_resid.mean() - slope * t_years.mean())
        trend_pct = (np.exp(intercept + slope * t_years) - 1.0) * 100.0
        ax.plot(dates, trend_pct, color=MODEL_COLOR, linewidth=1.8,
                label=f"trend {rate:+.1f}%/yr", zorder=2)
        ax.legend(fontsize=8)

    ax.set_xlabel("Gauging date")
    ax.set_ylabel("Observed − modelled (%)")
    ax.set_title("Rating-curve residuals over time")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig
