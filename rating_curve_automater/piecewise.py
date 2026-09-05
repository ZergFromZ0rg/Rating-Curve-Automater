"""Continuous piecewise power-law rating curve with data-driven segment count.

A single power law ``Q = a·(H − h0)^b`` is a straight line in log–log space. A
channel with more than one hydraulic control (low-flow notch, bank-full, over-
bank) bends: its log–log plot is better described by a few joined straight
segments. This module fits that as a **continuous** linear spline in
``(ln(H − h0), ln Q)`` space — knots (breakpoints) are chosen by forward
selection and the number of segments is picked by an information criterion
(BIC by default), so the fit does not gain segments it cannot justify.

The public entry point is :func:`fit_piecewise_power_law`; :func:`evaluate_spline`
turns the returned model into discharge at any stage and is what
``rating_curve_fitting.predict_discharge`` calls for a segmented fit.
"""

from __future__ import annotations

import math

import numpy as np

#: A segment must contain at least this many gaugings and span >= 2 distinct
#: stages (otherwise its slope is not identifiable).
MIN_SEGMENT_POINTS = 4
#: ...and at least this fraction of all gaugings.
MIN_SEGMENT_FRACTION = 0.10
#: Most segments ``fit_piecewise_power_law(..., n_segments=None)`` will consider.
DEFAULT_MAX_SEGMENTS = 4
_TINY = 1e-9

#: When ``fit_piecewise_power_law`` picks the segment count *and* the caller
#: estimated ``h0`` from the data (``refine_h0=True``), the single-power-law
#: baseline is re-profiled within +/- this window (metres) of the incoming
#: estimate to the ``h0`` that minimises its log-space RSS. This gives one power
#: law the same freedom a knot implicitly grants: a small ``h0`` error bends
#: ``ln Q`` vs ``ln(H - h0)``, and without this the forward search "fixes" the
#: bend with a spurious knot that BIC then accepts. A single power law's log-RSS
#: has a sharp interior minimum at the true ``h0`` when the data really is one
#: power law and collapses toward the datum when it is not, so the search is
#: bounded and a rail there simply lets a genuine knot win.
H0_REFINE_WINDOW_M = 0.10


def segment_r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Ordinary R² of a segment's gaugings, or ``nan`` when it has too few
    points / no spread to define one."""
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    if obs.size < 2 or np.ptp(obs) == 0:
        return float("nan")
    ss_res = float(np.sum((obs - pred) ** 2))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot else 1.0


def _design(u: np.ndarray, knots_u: list[float]) -> np.ndarray:
    cols = [np.ones_like(u), u]
    for k in knots_u:
        cols.append(np.maximum(u - k, 0.0))
    return np.column_stack(cols)


def fit_spline_coef(
    u: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None,
    knots_u: list[float],
) -> np.ndarray:
    """Weighted least-squares coefficients of the log-space linear spline.

    ``weights`` are applied exactly as :func:`numpy.polyfit`'s ``w`` (they scale
    the residual, not its square), so a segmented fit and a single-segment fit
    weight their gaugings identically.
    """
    x = _design(u, knots_u)
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        x = x * w[:, None]
        y = y * w
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return coef


def evaluate_spline(spline: dict, stage) -> np.ndarray:
    """Discharge from a fitted spline at ``stage`` (scalar or array)."""
    stage = np.atleast_1d(np.asarray(stage, dtype=float))
    u = np.log(np.maximum(stage - spline["h0"], _TINY))
    coef = spline["coef"]
    y = coef[0] + coef[1] * u
    for j, k in enumerate(spline["knots_u"]):
        y = y + coef[2 + j] * np.maximum(u - k, 0.0)
    return np.exp(y)


def _segment_bounds(knots_u: list[float], u_lo: float, u_hi: float) -> list[tuple[float, float]]:
    edges = [u_lo, *knots_u, u_hi]
    return list(zip(edges[:-1], edges[1:]))


def _local_power_law(coef: np.ndarray, knots_u: list[float], seg_index: int) -> tuple[float, float]:
    """``(a_k, b_k)`` such that ``Q = a_k·(H − h0)^b_k`` on segment ``seg_index``."""
    b = coef[1] + sum(coef[2 + j] for j in range(seg_index))
    ln_a = coef[0] - sum(coef[2 + j] * knots_u[j] for j in range(seg_index))
    return float(math.exp(ln_a)), float(b)


def _weighted_rss_log(x_design: np.ndarray, y: np.ndarray, weights: np.ndarray | None, coef: np.ndarray) -> float:
    resid = y - x_design @ coef
    if weights is not None:
        resid = resid * np.asarray(weights, dtype=float)
    return float(np.sum(resid ** 2))


def _criterion(rss: float, n: int, n_segments: int, kind: str) -> float:
    n_params = 2 * n_segments  # 2 free params per segment (knots counted)
    if rss <= 0:
        rss = 1e-300
    ll_term = n * math.log(rss / n)
    penalty = n_params * math.log(n) if kind == "bic" else 2 * n_params
    return ll_term + penalty


def _valid_partition(sorted_stage: np.ndarray, knot_stages: list[float], min_points: int) -> bool:
    """Every segment (split by ``stage < knot``) has >= ``min_points`` gaugings
    across >= 2 distinct stages."""
    edges = [-np.inf, *sorted(knot_stages), np.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        seg = sorted_stage[(sorted_stage >= lo) & (sorted_stage < hi)]
        if len(seg) < min_points or np.unique(seg).size < 2:
            return False
    return True


def _best_single_h0(
    stage_sorted: np.ndarray,
    discharge_sorted: np.ndarray,
    weights_sorted: np.ndarray | None,
    h0_0: float,
    lo_b: float,
    hi_b: float,
) -> tuple[float, np.ndarray, float]:
    """``h0`` in ``[lo_b, hi_b]`` minimising the weighted single-power-law
    log-space RSS, with its refitted coefficients and that RSS.

    Used only for the ``auto`` segment-count baseline (see
    :data:`H0_REFINE_WINDOW_M`). ``stage_sorted`` is already ascending and masked
    to ``stage > h0`` at the incoming estimate; ``hi_b`` is kept below the lowest
    stage by the caller so every candidate keeps ``ln(H - h0)`` finite.
    """
    y = np.log(discharge_sorted)
    best_h0, best_coef, best_rss = float(h0_0), None, math.inf
    for cand in np.linspace(lo_b, hi_b, 65):
        u = np.log(stage_sorted - cand)
        coef = fit_spline_coef(u, y, weights_sorted, [])
        rss = _weighted_rss_log(_design(u, []), y, weights_sorted, coef)
        if rss < best_rss:
            best_h0, best_coef, best_rss = float(cand), coef, rss
    if best_coef is None:  # pragma: no cover - lo_b == hi_b
        u = np.log(stage_sorted - h0_0)
        best_coef = fit_spline_coef(u, y, weights_sorted, [])
        best_rss = _weighted_rss_log(_design(u, []), y, weights_sorted, best_coef)
    return best_h0, best_coef, best_rss


def fit_piecewise_power_law(
    stage: np.ndarray,
    discharge: np.ndarray,
    h0: float,
    weights: np.ndarray | None = None,
    *,
    n_segments: int | None = None,
    max_segments: int = DEFAULT_MAX_SEGMENTS,
    criterion: str = "bic",
    refine_h0: bool = False,
) -> dict:
    """Fit a continuous piecewise power law.

    ``n_segments`` fixes the segment count (>= 2); ``None`` lets ``criterion``
    ("bic" or "aic") choose it up to ``max_segments`` by forward knot selection.
    Returns a dict with ``spline`` (for :func:`evaluate_spline`), ``segments``
    (per-segment ``a`` / ``b`` / range / point count), ``breakpoints`` (stage
    values), ``n_segments``, log-space ``rss`` / ``criterion`` and the ``h0``
    the returned model uses.

    ``refine_h0`` (auto count only) re-profiles ``h0`` for the single-power-law
    baseline that knots must beat — see :data:`H0_REFINE_WINDOW_M`. Pass it when
    ``h0`` was estimated from the data, not supplied.
    """
    if criterion not in ("bic", "aic"):
        raise ValueError("criterion must be 'bic' or 'aic'.")

    stage = np.asarray(stage, dtype=float)
    discharge = np.asarray(discharge, dtype=float)
    keep = (stage - h0 > 0) & (discharge > 0)
    stage_m, discharge_m = stage[keep], discharge[keep]
    weights_m = np.asarray(weights, dtype=float)[keep] if weights is not None else None

    order = np.argsort(stage_m, kind="stable")
    stage_m, discharge_m = stage_m[order], discharge_m[order]
    weights_m = weights_m[order] if weights_m is not None else None

    n = stage_m.size
    u = np.log(stage_m - h0)
    y = np.log(discharge_m)
    u_lo, u_hi = float(u[0]), float(u[-1])

    min_points = max(MIN_SEGMENT_POINTS, int(MIN_SEGMENT_FRACTION * n))
    forced = n_segments is not None
    if forced and n_segments < 2:
        raise ValueError("n_segments must be >= 2 (use the single power law otherwise).")
    target_knots = (n_segments - 1) if forced else max(1, max_segments - 1)

    interior = np.unique(stage_m)[1:-1]

    def evaluate(knot_stages: list[float]) -> tuple[np.ndarray, float]:
        knots_u = [math.log(s - h0) for s in sorted(knot_stages)]
        design = _design(u, knots_u)
        coef = fit_spline_coef(u, y, weights_m, knots_u)
        rss = _weighted_rss_log(design, y, weights_m, coef)
        return coef, rss

    knot_stages: list[float] = []
    coef0, rss0 = evaluate(knot_stages)
    best = {"knot_stages": [], "coef": coef0, "rss": rss0,
            "crit": _criterion(rss0, n, 1, criterion), "h0": float(h0)}

    # Give the single-power-law baseline the same h0 freedom a knot implicitly
    # grants, so the forward search does not "fix" a small h0 error with a
    # spurious knot. Only when h0 was estimated and the count is not forced.
    if refine_h0 and not forced:
        lo_b = max(_TINY, h0 - H0_REFINE_WINDOW_M)
        hi_b = min(float(stage_m[0]) - _TINY, h0 + H0_REFINE_WINDOW_M)
        if hi_b > lo_b:
            h0_ref, coef_ref, rss_ref = _best_single_h0(
                stage_m, discharge_m, weights_m, h0, lo_b, hi_b
            )
            crit_ref = _criterion(rss_ref, n, 1, criterion)
            if crit_ref < best["crit"]:
                best = {"knot_stages": [], "coef": coef_ref, "rss": rss_ref,
                        "crit": crit_ref, "h0": h0_ref}

    while len(knot_stages) < target_knots:
        trial = None
        for s in interior:
            if s in knot_stages:
                continue
            cand = sorted([*knot_stages, float(s)])
            if not _valid_partition(stage_m, cand, min_points):
                continue
            coef, rss = evaluate(cand)
            crit = _criterion(rss, n, len(cand) + 1, criterion)
            if trial is None or rss < trial["rss"]:
                trial = {"knot_stages": cand, "coef": coef, "rss": rss, "crit": crit}
        if trial is None:
            break
        if not forced and trial["crit"] >= best["crit"] - 1e-9:
            break
        knot_stages = trial["knot_stages"]
        best = trial

    if forced and len(best["knot_stages"]) != n_segments - 1:
        raise ValueError(
            f"Could not place {n_segments - 1} breakpoint(s) with >= {min_points} "
            f"gaugings per segment; use fewer segments."
        )

    h0_used = float(best.get("h0", h0))
    knots_u = [math.log(s - h0_used) for s in best["knot_stages"]]
    coef = best["coef"]
    k = len(knots_u) + 1

    if k == 1:
        a, b = _local_power_law(coef, [], 0)
        return {
            "is_segmented": False,
            "spline": {"h0": h0_used, "knots_u": [], "coef": [float(c) for c in coef]},
            "a": a,
            "b": b,
            "n_segments": 1,
            "h0": h0_used,
            "rss_log": best["rss"],
            "criterion": criterion,
            "criterion_value": best["crit"],
        }

    spline = {"h0": h0_used, "knots_u": [float(x) for x in knots_u], "coef": [float(c) for c in coef]}
    modeled = evaluate_spline(spline, stage_m)
    bounds = _segment_bounds(knots_u, u_lo, u_hi + _TINY)

    segments = []
    for i, (lo, hi) in enumerate(bounds):
        in_seg = (u >= lo) & (u <= hi) if i == len(bounds) - 1 else (u >= lo) & (u < hi)
        a_k, b_k = _local_power_law(coef, knots_u, i)
        segments.append({
            "a": a_k,
            "b": b_k,
            "r_squared": segment_r_squared(discharge_m[in_seg], modeled[in_seg]),
            "n_points": int(in_seg.sum()),
            "stage_min": float(math.exp(lo) + h0),
            "stage_max": float(math.exp(min(hi, u_hi)) + h0),
        })

    return {
        "is_segmented": True,
        "spline": spline,
        "segments": segments,
        "breakpoints": [float(s) for s in best["knot_stages"]],
        "n_segments": k,
        "h0": h0_used,
        "rss_log": best["rss"],
        "criterion": criterion,
        "criterion_value": best["crit"],
        # back-compat scalars describe the lowest segment
        "a": segments[0]["a"],
        "b": segments[0]["b"],
    }
