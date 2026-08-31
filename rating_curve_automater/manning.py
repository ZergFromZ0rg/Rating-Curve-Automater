"""Hydraulic sanity check for a fitted rating curve.

A power-law rating curve is only supported by data *within* the range of the
gaugings. Above the highest gauging it is a blind extrapolation whose slope is
whatever the top few points happened to imply. This module computes an
independent, physically-based curve from a **surveyed cross-section** using
Manning's uniform-flow equation

    Q = (1 / n) * A * R**(2/3) * S**(1/2)          (SI, k_n = 1)

    A = wetted area, P = wetted perimeter (bed contact only), R = A / P,
    S = energy slope (≈ bed slope for uniform flow), n = Manning roughness.

``n`` is calibrated to the rating over the gauged range (so the check is about
the *shape* of the extrapolation, not a guessed roughness), unless the caller
supplies one. :func:`manning_sanity_check` then flags where the fitted curve and
the Manning curve disagree, weighting the extrapolated stages.

Assumptions and limits, stated plainly:

* the cross-section elevations and the rating's stage ``H`` share a vertical
  datum (pass ``stage_offset`` if water-surface elevation = ``H + offset``);
* a single ``n`` for the whole section -- a wide vegetated floodplain with the
  same ``n`` as the channel will over-predict, so the check is most trustworthy
  for in-bank and slightly-out-of-bank extrapolation;
* steady uniform flow (no backwater, no hysteresis).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_OFFSET_ALIASES = ("offset", "station", "distance", "chainage", "x", "dist", "stn")
_ELEV_ALIASES = ("elevation", "elev", "bed", "bed_elevation", "z", "ground", "level", "y")


def read_cross_section(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a cross-section CSV into ``(offset, bed_elevation)`` arrays.

    Looks for an offset/station column and an elevation/bed column by name
    (case-insensitive); falls back to the first two numeric columns. Rows are
    sorted by offset.
    """
    df = pd.read_csv(path)
    cols = {str(c).strip().lower(): c for c in df.columns}

    def _pick(aliases: tuple[str, ...]) -> str | None:
        for a in aliases:
            if a in cols:
                return cols[a]
        for lc, orig in cols.items():
            if any(a in lc for a in aliases):
                return orig
        return None

    off_col, elev_col = _pick(_OFFSET_ALIASES), _pick(_ELEV_ALIASES)
    if off_col is None or elev_col is None or off_col == elev_col:
        numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric) < 2:
            raise ValueError(
                "Cross-section CSV needs an offset/station column and an "
                "elevation/bed column (or at least two numeric columns)."
            )
        off_col, elev_col = numeric[:2]

    out = df[[off_col, elev_col]].apply(pd.to_numeric, errors="coerce").dropna()
    out = out.sort_values(off_col)
    return out[off_col].to_numpy(dtype=float), out[elev_col].to_numpy(dtype=float)

#: Manning's n outside this range is physically implausible for a natural
#: channel -- if the value calibrated to the rating lands outside it, the rating
#: (or the survey, or the slope) is suspect.
PLAUSIBLE_N = (0.015, 0.20)
#: Divergence thresholds (percent) between the fitted curve and the calibrated
#: Manning curve, over the *extrapolated* stage range, for the flag.
_DIVERGE_CHECK_PCT = 15.0
_DIVERGE_BAD_PCT = 35.0
_MIN_SECTION_POINTS = 4


def section_hydraulics(offset: np.ndarray, bed_elev: np.ndarray, wse: float) -> tuple[float, float, float]:
    """``(area, wetted_perimeter, top_width)`` of a cross-section at water-surface
    elevation ``wse``.

    ``offset`` (m across the section, increasing) and ``bed_elev`` (m, same
    datum as ``wse``) describe the ground line. Handles a section that dips
    below ``wse`` in more than one place (multiple channels). The water surface
    is not part of the wetted perimeter.
    """
    x = np.asarray(offset, dtype=float)
    z = np.asarray(bed_elev, dtype=float)
    area = perim = width = 0.0

    for i in range(x.size - 1):
        x0, x1 = float(x[i]), float(x[i + 1])
        z0, z1 = float(z[i]), float(z[i + 1])
        d0, d1 = wse - z0, wse - z1
        dx = x1 - x0
        if dx < 0:
            continue
        if dx == 0:                                # vertical wall: wetted height only
            z_lo, z_hi = sorted((z0, z1))
            perim += max(0.0, min(wse, z_hi) - z_lo)
            continue

        if d0 <= 0 and d1 <= 0:
            continue
        if d0 > 0 and d1 > 0:                       # fully wetted segment
            area += 0.5 * (d0 + d1) * dx
            perim += np.hypot(dx, z1 - z0)
            width += dx
            continue

        # Partially wetted: the ground line crosses wse inside the segment.
        frac = d0 / (d0 - d1)                       # in (0, 1); share from x0
        xc = x0 + frac * dx
        if d0 > 0:                                  # wet on the left, dry right
            area += 0.5 * d0 * (xc - x0)
            perim += np.hypot(xc - x0, wse - z0)
            width += xc - x0
        else:                                       # dry left, wet on the right
            area += 0.5 * d1 * (x1 - xc)
            perim += np.hypot(x1 - xc, wse - z1)
            width += x1 - xc

    return area, perim, width


def manning_q(area: float, perimeter: float, slope: float, n: float) -> float:
    """Manning discharge for a section with the given wetted geometry."""
    if area <= 0 or perimeter <= 0 or slope <= 0 or n <= 0:
        return 0.0
    r = area / perimeter
    return (1.0 / n) * area * r ** (2.0 / 3.0) * slope ** 0.5


def _n_for_q(area: float, perimeter: float, slope: float, q_target: float) -> float | None:
    """The ``n`` that makes :func:`manning_q` equal ``q_target`` for this geometry."""
    if area <= 0 or perimeter <= 0 or slope <= 0 or q_target <= 0:
        return None
    r = area / perimeter
    return area * r ** (2.0 / 3.0) * slope ** 0.5 / q_target


def calibrate_n(
    fit: dict,
    offset: np.ndarray,
    bed_elev: np.ndarray,
    slope: float,
    *,
    stage_offset: float = 0.0,
    n_calibration_stages: np.ndarray | None = None,
) -> float | None:
    """Median of the ``n`` values that match the rating at each gauged stage.

    ``n_calibration_stages`` defaults to a grid over the gauged range
    (``fit["stage_min"]`` .. ``fit["stage_max"]``).
    """
    from rating_curve_automater.rating_curve_fitting import predict_discharge

    lo = float(fit.get("stage_min", np.min(bed_elev)))
    hi = float(fit.get("stage_max", lo + 1.0))
    if n_calibration_stages is None:
        n_calibration_stages = np.linspace(lo + 0.02 * (hi - lo), hi, 12)

    values: list[float] = []
    for h in np.atleast_1d(n_calibration_stages):
        area, perim, _ = section_hydraulics(offset, bed_elev, float(h) + stage_offset)
        q_rating = float(predict_discharge(fit, float(h))[0])
        n_h = _n_for_q(area, perim, slope, q_rating)
        if n_h is not None and np.isfinite(n_h):
            values.append(n_h)
    if not values:
        return None
    return float(np.median(values))


def manning_sanity_check(
    fit: dict,
    offset,
    bed_elev,
    slope: float,
    *,
    n: float | None = None,
    stage_offset: float = 0.0,
    extrapolate_to: float | None = None,
    n_points: int = 60,
) -> dict:
    """Compare the fitted rating curve with a Manning curve from a cross-section.

    ``offset`` / ``bed_elev`` are the surveyed ground line (m; same vertical
    datum as stage, or pass ``stage_offset`` so WSE = ``H + stage_offset``).
    ``slope`` is the channel/energy slope (m/m). ``n`` overrides the roughness
    that is otherwise calibrated to the rating over the gauged range.
    ``extrapolate_to`` caps the check stage (default: the lowest surveyed bank
    top, i.e. as high as the section can carry a defined water surface).

    Returns a dict with the two curves on a common grid, their percent
    difference, the calibrated ``n``, and a ``flag`` of ``"ok"`` / ``"check"`` /
    ``"diverges"`` / ``"implausible-n"`` / ``"unusable"`` plus a ``message``.
    """
    from rating_curve_automater.rating_curve_fitting import predict_discharge

    offset = np.asarray(offset, dtype=float)
    bed_elev = np.asarray(bed_elev, dtype=float)
    order = np.argsort(offset)
    offset, bed_elev = offset[order], bed_elev[order]

    result: dict = {"flag": "unusable", "n_supplied": n, "slope": float(slope),
                    "stage_offset": float(stage_offset)}

    if offset.size < _MIN_SECTION_POINTS or np.ptp(offset) <= 0:
        result["message"] = "Cross-section needs at least 4 points spanning a positive width."
        return result
    if not slope > 0:
        result["message"] = "Channel slope must be positive."
        return result

    stage_min = float(fit.get("stage_min", bed_elev.min() - stage_offset))
    stage_max = float(fit.get("stage_max", stage_min + 1.0))
    bank_top = float(bed_elev[1:-1].max() if offset.size > 2 else bed_elev.max()) - stage_offset
    ceiling = extrapolate_to if extrapolate_to is not None else min(bank_top, stage_max * 2.0)
    ceiling = max(ceiling, stage_max + 1e-6)

    n_calibrated = calibrate_n(fit, offset, bed_elev, slope, stage_offset=stage_offset)
    result["n_calibrated"] = n_calibrated
    n_used = n if n is not None else n_calibrated
    if n_used is None or not np.isfinite(n_used) or n_used <= 0:
        result["message"] = (
            "Could not calibrate Manning's n to the rating (check the slope and "
            "that the cross-section and stage share a datum)."
        )
        return result
    result["n_used"] = float(n_used)

    grid = np.linspace(max(stage_min, bed_elev.min() - stage_offset + 1e-6), ceiling, n_points)
    q_rating = np.asarray(predict_discharge(fit, grid), dtype=float)
    q_manning = np.array([
        manning_q(*section_hydraulics(offset, bed_elev, float(h) + stage_offset)[:2], slope, n_used)
        for h in grid
    ])

    with np.errstate(divide="ignore", invalid="ignore"):
        pct_diff = np.where(q_manning > 0, (q_rating - q_manning) / q_manning * 100.0, np.nan)

    extrap = grid > stage_max
    gauged = ~extrap
    result.update({
        "stage": grid,
        "q_rating": q_rating,
        "q_manning": q_manning,
        "pct_diff": pct_diff,
        "stage_max_gauged": stage_max,
        "extrapolation_ceiling": float(ceiling),
        "max_abs_pct_diff_gauged": float(np.nanmax(np.abs(pct_diff[gauged]))) if gauged.any() else float("nan"),
        "max_abs_pct_diff_extrapolated": float(np.nanmax(np.abs(pct_diff[extrap]))) if extrap.any() else float("nan"),
    })

    n_plausible = PLAUSIBLE_N[0] <= n_used <= PLAUSIBLE_N[1]
    worst_extrap = result["max_abs_pct_diff_extrapolated"]

    if not n_plausible:
        result["flag"] = "implausible-n"
        result["message"] = (
            f"Manning's n calibrated to the rating is {n_used:.3f} — outside the "
            f"{PLAUSIBLE_N[0]:.3f}–{PLAUSIBLE_N[1]:.2f} range expected for a natural "
            f"channel. The rating slope, the surveyed section or the channel slope "
            f"is likely off."
        )
    elif not np.isfinite(worst_extrap) or not extrap.any():
        result["flag"] = "ok"
        result["message"] = (
            f"Manning's n = {n_used:.3f} (calibrated). No stages above the highest "
            f"gauging to check."
        )
    elif worst_extrap >= _DIVERGE_BAD_PCT:
        result["flag"] = "diverges"
        result["message"] = (
            f"The fitted curve and the cross-section (Manning, n = {n_used:.3f}) "
            f"disagree by up to {worst_extrap:.0f}% above the highest gauging "
            f"({stage_max:.2f} m). The power-law extrapolation is not supported by "
            f"the channel geometry — do not rely on it; survey a high-flow gauging "
            f"or use the Manning curve above {stage_max:.2f} m."
        )
    elif worst_extrap >= _DIVERGE_CHECK_PCT:
        result["flag"] = "check"
        result["message"] = (
            f"The fitted curve drifts {worst_extrap:.0f}% from the cross-section "
            f"(Manning, n = {n_used:.3f}) by {ceiling:.2f} m — treat the "
            f"extrapolation above {stage_max:.2f} m with caution."
        )
    else:
        result["flag"] = "ok"
        result["message"] = (
            f"The fitted curve tracks the cross-section (Manning, n = {n_used:.3f}) "
            f"to within {worst_extrap:.0f}% up to {ceiling:.2f} m — the "
            f"extrapolation is consistent with the channel geometry."
        )
    return result
