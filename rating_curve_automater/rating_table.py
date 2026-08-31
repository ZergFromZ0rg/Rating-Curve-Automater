"""Build a stage → discharge lookup table from a fitted rating curve.

This is the artifact a hydrographer actually applies to a continuous stage
record: discharge tabulated at a fixed stage increment, with the bootstrap
confidence / prediction bounds carried alongside and any rows outside the
gauged stage range flagged as extrapolation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_STAGE_STEP_M = 0.01
_MAX_ROWS = 200_000

OUT_STAGE = "Stage (m)"
OUT_Q = "Discharge (m³/s)"
OUT_IN_RANGE = "Within gauged range"


def build_rating_table(
    fit: dict,
    *,
    step: float = DEFAULT_STAGE_STEP_M,
    stage_min: float | None = None,
    stage_max: float | None = None,
) -> pd.DataFrame:
    """Tabulate ``Q`` against stage on a fixed grid.

    ``step`` is the stage increment (m). ``stage_min`` / ``stage_max`` default to
    the gauged stage range recorded on ``fit`` (``stage_min`` / ``stage_max``);
    widening them past that range produces extrapolated rows, flagged ``False``
    in the *Within gauged range* column. Confidence / prediction columns are
    filled by interpolating ``fit["bands"]`` and left blank outside the
    bootstrap grid.
    """
    from rating_curve_automater.rating_curve_fitting import predict_discharge

    if step <= 0:
        raise ValueError("step must be positive.")

    h0 = float(fit["h0"])
    gauged_min = fit.get("stage_min")
    gauged_max = fit.get("stage_max")

    lo = stage_min if stage_min is not None else gauged_min
    hi = stage_max if stage_max is not None else gauged_max
    if lo is None or hi is None:
        raise ValueError(
            "No stage range: pass stage_min/stage_max, or a fit that records stage_min/stage_max."
        )
    if hi <= lo:
        raise ValueError(f"stage_max ({hi}) must exceed stage_min ({lo}).")

    start = np.floor(lo / step) * step
    stop = np.ceil(hi / step) * step
    if (stop - start) / step > _MAX_ROWS:
        raise ValueError(
            f"{(stop - start) / step:.0f} rows requested; raise `step` or narrow the range."
        )

    stages = np.round(np.arange(start, stop + step / 2, step), 9)
    # Q is only defined above the stage of zero flow.
    stages = stages[stages > h0]

    q = predict_discharge(fit, stages) if stages.size else np.array([])
    out = pd.DataFrame({OUT_STAGE: stages, OUT_Q: q})

    bands = fit.get("bands")
    if bands and stages.size:
        grid = np.asarray(bands["stage"], dtype=float)
        within_grid = (stages >= grid.min()) & (stages <= grid.max())
        pct = int(round(bands["level"] * 100))
        for label, key in (
            (f"{pct}% confidence lower", "ci_lower"),
            (f"{pct}% confidence upper", "ci_upper"),
            (f"{pct}% prediction lower", "pi_lower"),
            (f"{pct}% prediction upper", "pi_upper"),
        ):
            col = np.interp(stages, grid, np.asarray(bands[key], dtype=float))
            col[~within_grid] = np.nan
            out[label] = col

    if gauged_min is not None and gauged_max is not None and stages.size:
        out[OUT_IN_RANGE] = (stages >= gauged_min - 1e-9) & (stages <= gauged_max + 1e-9)

    return out


def export_rating_table_csv(
    fit: dict,
    output_path: str | Path,
    *,
    step: float = DEFAULT_STAGE_STEP_M,
    stage_min: float | None = None,
    stage_max: float | None = None,
) -> Path:
    """Write :func:`build_rating_table` to CSV and return the path."""
    table = build_rating_table(fit, step=step, stage_min=stage_min, stage_max=stage_max)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    return output
