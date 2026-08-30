"""Matplotlib rating-curve figure, shared by the GUI preview."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.rating_curve_fitting import (
    DISCHARGE_COL,
    STAGE_COL,
    predict_discharge,
    select_valid_measurements,
)

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
    stage = working[STAGE_COL].to_numpy(dtype=float)
    observed = working[DISCHARGE_COL].to_numpy(dtype=float)

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

    curve_stage = np.linspace(float(stage.min()), float(stage.max()), 300)
    curve_stage = curve_stage[curve_stage > h0]
    if fit is not None and fit.get("is_segmented"):
        modeled = predict_discharge(fit, curve_stage)
        label = f"Segmented (break H={fit['breakpoint']:.3f})"
    else:
        modeled = a * np.power(curve_stage - h0, b)
        label = f"Q = {a:.3f}·(H−{h0:.3f})^{b:.3f}"
    ax.plot(curve_stage, modeled, color=MODEL_COLOR, linewidth=2, label=label, zorder=2)
    if fit is not None and fit.get("is_segmented"):
        ax.axvline(fit["breakpoint"], color="#7f7f7f", linestyle="--", linewidth=1, zorder=1)

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
