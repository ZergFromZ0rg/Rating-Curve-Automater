from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rating_curve_automater.piecewise import (
    DEFAULT_MAX_SEGMENTS,
    evaluate_spline,
    fit_piecewise_power_law,
    fit_spline_coef,
)
from rating_curve_automater.rating_curve_uncertainty import DEFAULT_N_BOOTSTRAP
from rating_curve_automater.schema import DISCHARGE_CMS, DISCHARGE_UNCERTAINTY, STAGE_M, ensure_canonical

DEFAULT_H0 = 0.18

#: Assumed relative uncertainty (percent of Q) for discharge measurements that
#: carry no per-point uncertainty. Used to weight the fit and, later, to size
#: confidence bands. A uniform value leaves the fit unchanged from plain OLS;
#: only a varying per-point column actually re-weights the regression.
DEFAULT_DISCHARGE_UNCERTAINTY_PCT = 7.0

# Back-compat aliases (canonical names now live in rating_curve_automater.schema).
STAGE_COL = STAGE_M
DISCHARGE_COL = DISCHARGE_CMS


def select_valid_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Return the stage/discharge rows that are usable for curve work.

    Accepts a canonical or a raw frame (see :func:`~rating_curve_automater.schema.ensure_canonical`),
    applies the ``is_valid`` flag when present, and drops rows missing a stage
    or discharge value. Used by both the fitter and the report so the two always
    operate on the same set of points.
    """
    working = ensure_canonical(df, required=(STAGE_M, DISCHARGE_CMS)).copy()
    if "is_valid" in working.columns:
        working = working[working["is_valid"].fillna(False).astype(bool)]
    working = working.dropna(subset=[STAGE_M, DISCHARGE_CMS])
    return working.copy()


def _loglog_fit(
    stage: np.ndarray,
    discharge: np.ndarray,
    h0: float,
    weights: np.ndarray | None = None,
) -> dict | None:
    """Fit ``Q = a * (H - h0)^b`` by linear regression in log-log space.

    ``weights`` (one per point, ``1 / fractional-uncertainty``) turns the
    regression into a weighted least-squares fit: gaugings with a larger stated
    measurement uncertainty pull the curve less. ``None`` (or a constant array)
    is an ordinary least-squares fit.

    Returns ``None`` when fewer than two points have ``H - h0 > 0``.
    """
    x = stage - h0
    mask = x > 0
    if int(mask.sum()) < 2:
        return None

    discharge_m = discharge[mask]
    x_m = x[mask]
    w_m = weights[mask] if weights is not None else None

    slope, intercept = np.polyfit(np.log(x_m), np.log(discharge_m), 1, w=w_m)
    a = float(np.exp(intercept))
    b = float(slope)

    predicted = a * np.power(x_m, b)
    residuals = discharge_m - predicted
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((discharge_m - np.mean(discharge_m)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot != 0 else 1.0

    # Weighted R² (real space): the same goodness-of-fit measure but judged on
    # the points the regression actually cared about, so a deliberately
    # down-weighted outlier does not drag it down.
    if w_m is not None:
        pw = np.asarray(w_m, dtype=float) ** 2
        q_bar_w = float(np.sum(pw * discharge_m) / np.sum(pw))
        ss_res_w = float(np.sum(pw * residuals ** 2))
        ss_tot_w = float(np.sum(pw * (discharge_m - q_bar_w) ** 2))
        r_squared_weighted = 1.0 - ss_res_w / ss_tot_w if ss_tot_w != 0 else 1.0
    else:
        r_squared_weighted = float(r_squared)

    return {
        "a": a,
        "b": b,
        "r_squared": float(r_squared),
        "r_squared_weighted": float(r_squared_weighted),
        "ss_res": ss_res,
        "n_points": int(mask.sum()),
    }


#: ``h0`` estimation: the classic three-point method anchors it (robust, never
#: runs away); the log-log curvature method then refines within a window of
#: +/- :data:`_H0_REFINE_WINDOW_M` around that anchor, using the lowest
#: :data:`_H0_LOWFLOW_FRACTION` of gaugings (the low-flow control is what the
#: point of zero flow depends on). The refinement is only trusted when the block
#: actually straightens (|curvature| < :data:`_H0_STRAIGHT_TOL`).
_H0_MIN_POINTS = 10
_H0_LOWFLOW_FRACTION = 0.45
_H0_REFINE_WINDOW_M = 0.08
_H0_STRAIGHT_TOL = 0.07
#: Three-point triples disagreeing by more than this fraction of the stage range
#: (inter-quartile) => the anchor is unreliable; fall back to a neutral default.
_H0_TRIPLE_IQR_FRACTION = 0.12


def _weighted_polyfit(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None, deg: int) -> np.ndarray:
    """``numpy.polyfit``-style coefficients (highest power first), weights scaling
    the residual as :func:`numpy.polyfit`'s ``w`` does."""
    vander = np.vander(np.asarray(x, dtype=float), deg + 1)
    y = np.asarray(y, dtype=float)
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        vander = vander * w[:, None]
        y = y * w
    coef, *_ = np.linalg.lstsq(vander, y, rcond=None)
    return coef


def _log_curvature(stage, discharge, h0: float, weights=None) -> float:
    """Quadratic coefficient of ``ln Q`` regressed on ``ln(H - h0)``.

    Zero means the gaugings fall on a straight line in log-log space, i.e. a
    single power law with this ``h0`` describes them. A wrong ``h0`` bends the
    line: too small curves it up, too large curves it down.
    """
    x = np.asarray(stage, dtype=float) - h0
    mask = x > 1e-9
    if int(mask.sum()) < 6:
        return float("nan")
    y = np.log(np.asarray(discharge, dtype=float)[mask])
    w = None if weights is None else np.asarray(weights, dtype=float)[mask]
    return float(_weighted_polyfit(np.log(x[mask]), y, w, 2)[0])


def _three_point_h0(stage: np.ndarray, discharge: np.ndarray) -> tuple[float, float]:
    """Classic graphical point-of-zero-flow estimate, robustified.

    For three discharges in geometric progression the corresponding stages give
    ``h0 = (h1*h3 - h2^2) / (h1 + h3 - 2*h2)``. Many triples spanning the lower
    half of the flow range are solved -- staying low keeps them inside a single
    hydraulic control. Returns ``(median, iqr)``; a wide inter-quartile range
    means the triples disagree and the estimate should not be trusted.
    """
    order = np.argsort(stage)
    h = np.asarray(stage, dtype=float)[order]
    ln_q = np.log(np.asarray(discharge, dtype=float)[order])
    lo, hi = float(ln_q.min()), float(np.percentile(ln_q, 55))
    estimates: list[float] = []
    for frac in np.linspace(0.15, 0.85, 15):
        mid = lo + frac * (hi - lo)
        delta = min(mid - lo, hi - mid) * 0.9
        if delta <= 1e-6:
            continue
        h1, h2, h3 = np.interp([mid - delta, mid, mid + delta], ln_q, h)
        denom = h1 + h3 - 2.0 * h2
        if abs(denom) < 1e-9:
            continue
        estimates.append((h1 * h3 - h2 ** 2) / denom)
    if not estimates:
        return float("nan"), float("inf")
    arr = np.asarray(estimates)
    return float(np.median(arr)), float(np.subtract(*np.percentile(arr, [75, 25])))


def _estimate_h0(
    stage: np.ndarray,
    discharge: np.ndarray,
    bounds: tuple[float, float] | None = None,
    weights: np.ndarray | None = None,
) -> tuple[float, dict]:
    """Estimate the stage of zero flow and report how confident the estimate is.

    The classic three-point (geometric-progression) method anchors ``h0``. The
    log-log curvature method -- the ``h0`` that makes ``ln Q`` vs ``ln(H - h0)``
    straightest over the low-flow gaugings -- then refines it within a narrow
    window of that anchor, but only when those gaugings genuinely straighten.
    Neither step maximises R² (which is monotone toward ``h0 -> 0`` because
    ``a``, ``b`` and ``h0`` trade off, so it collapses the estimate).

    Returns ``(h0, diagnostics)`` with ``method`` (``three-point`` /
    ``curvature`` / ``default``), ``railed`` (hit a search bound -- weakly
    identified) and ``min_abs_curvature`` (how straight the low-flow block gets).
    """
    stage = np.asarray(stage, dtype=float)
    discharge = np.asarray(discharge, dtype=float)
    order = np.argsort(stage)
    stage, discharge = stage[order], discharge[order]
    w = None if weights is None else np.asarray(weights, dtype=float)[order]

    stage_min = float(stage.min())
    stage_range = float(stage.max() - stage_min)
    if bounds is None:
        # Without a user-supplied bound, keep h0 >= 0: a natural point of zero
        # flow below the datum is real but rare, and letting the estimate go
        # negative on noisy compound data is the main way a segmented fit blows
        # up. A genuine negative PZF site can pass ``bounds=`` / a fixed ``h0``.
        lo_b, hi_b = 0.0, 0.95 * stage_min
    else:
        lo_b, hi_b = bounds
        hi_b = min(hi_b, 0.999 * stage_min)
    diag = {"method": "three-point", "railed": False,
            "min_abs_curvature": float("nan"), "single_control": None}

    if not hi_b > lo_b or len(stage) < _H0_MIN_POINTS:
        return DEFAULT_H0, {**diag, "method": "default"}

    anchor, triple_iqr = _three_point_h0(stage, discharge)
    unreliable = not np.isfinite(anchor) or triple_iqr > _H0_TRIPLE_IQR_FRACTION * stage_range
    if unreliable:
        # Triples disagree -> no usable anchor. Use a neutral default, but never
        # above half-way to the lowest gauging.
        anchor = min(DEFAULT_H0, 0.5 * stage_min)
        diag["method"] = "default"
    anchor = float(np.clip(anchor, lo_b, hi_b))
    h0 = anchor

    # Refine within +/- window of the anchor, on the lowest gaugings only.
    k = min(len(stage), max(_H0_MIN_POINTS, int(_H0_LOWFLOW_FRACTION * len(stage))))
    lo_w = max(lo_b, anchor - _H0_REFINE_WINDOW_M)
    hi_w = min(hi_b, anchor + _H0_REFINE_WINDOW_M)
    best_h0, best_abs = float("nan"), np.inf
    for cand in np.linspace(lo_w, hi_w, 64):
        curv = _log_curvature(stage[:k], discharge[:k], float(cand),
                              None if w is None else w[:k])
        if np.isfinite(curv) and abs(curv) < best_abs:
            best_abs, best_h0 = abs(curv), float(cand)
    diag["min_abs_curvature"] = float(best_abs)
    if np.isfinite(best_h0) and best_abs <= _H0_STRAIGHT_TOL:
        h0 = best_h0
        diag["method"] = "curvature"

    h0 = float(np.clip(h0, lo_b, hi_b))
    # Pinned at 0 => the data wants h0 <= 0 (the a/b/h0 collapse) -- a real
    # problem. Pinned just below the lowest gauging with a straight low-flow fit
    # is normal (the gaugings simply don't bound h0 from above) and not flagged.
    diag["railed_low"] = bool(abs(h0 - lo_b) < 1e-6 and lo_b >= 0.0)
    diag["railed_high"] = bool(abs(h0 - hi_b) < 1e-6)
    diag["railed"] = bool(
        diag["railed_low"]
        or (diag["railed_high"] and diag["min_abs_curvature"] > _H0_STRAIGHT_TOL)
    )
    return h0, diag


def estimate_h0(
    stage: np.ndarray,
    discharge: np.ndarray,
    bounds: tuple[float, float] | None = None,
    weights: np.ndarray | None = None,
) -> float:
    """Estimate the stage of zero flow ``h0`` (see :func:`_estimate_h0`)."""
    return _estimate_h0(stage, discharge, bounds, weights)[0]


MIN_SEGMENT_POINTS = 4

#: Plausibility thresholds for :func:`assess_fit`.
MIN_POINTS_RELIABLE = 5
MIN_R_SQUARED = 0.5
MIN_ABS_LOG_CORR = 0.3


class ImplausibleRatingCurve(ValueError):
    """Raised by ``fit_rating_curve(..., strict=True)`` for a non-physical fit."""

    def __init__(self, warnings: list[str]) -> None:
        self.warnings = warnings
        super().__init__("; ".join(warnings))


def assess_fit(fit: dict, stage: np.ndarray, discharge: np.ndarray) -> tuple[list[str], bool]:
    """Return ``(warnings, has_critical)`` for a fitted curve.

    Critical = the relationship is not a rating curve at all (b <= 0, or stage
    and discharge uncorrelated). Non-critical = merely weak (low R², few points).
    """
    warnings_out: list[str] = []
    critical = False

    if fit.get("is_segmented"):
        b_value = min(seg["b"] for seg in fit["segments"])
    else:
        b_value = fit["b"]
    if b_value <= 0:
        warnings_out.append(
            f"fitted exponent b = {b_value:.3f} ≤ 0 — discharge does not increase "
            f"with stage; this is not a valid rating curve"
        )
        critical = True

    x = np.asarray(stage, dtype=float) - fit["h0"]
    mask = (x > 0) & (np.asarray(discharge, dtype=float) > 0)
    if int(mask.sum()) >= 3:
        with np.errstate(all="ignore"):
            corr = float(np.corrcoef(np.log(x[mask]), np.log(discharge[mask]))[0, 1])
        if np.isfinite(corr) and abs(corr) < MIN_ABS_LOG_CORR:
            warnings_out.append(
                f"stage and discharge are essentially uncorrelated (r = {corr:.2f})"
            )
            critical = True

    # Judge goodness of fit on the weighted R² when the fit was weighted, so a
    # deliberately down-weighted outlier is not counted against it.
    if fit.get("weighted"):
        r2_used = fit.get("r_squared_weighted", fit["r_squared"])
        r2_label = "weighted R²"
    else:
        r2_used = fit["r_squared"]
        r2_label = "R²"
    if r2_used < MIN_R_SQUARED:
        warnings_out.append(f"poor fit: {r2_label} = {r2_used:.2f}")
    if fit["n_points"] < MIN_POINTS_RELIABLE:
        warnings_out.append(f"only {fit['n_points']} point(s) — the fit is unreliable")

    if not fit.get("is_segmented") and fit["n_points"] >= 8:
        curv = _log_curvature(stage, discharge, float(fit["h0"]))
        if np.isfinite(curv) and abs(curv) > 0.10:
            shape = "concave" if curv > 0 else "convex"
            warnings_out.append(
                f"the gaugings are {shape} in log-log space (curvature = {curv:+.2f}) "
                f"— one power law misfits the mid-range; try segments=2 or "
                f"segments='auto'"
            )

    h0_diag = fit.get("h0_diagnostics")
    if h0_diag is not None:
        if h0_diag.get("railed"):
            warnings_out.append(
                f"h0 = {fit['h0']:.3f} m reached a search bound — the point of zero "
                f"flow is weakly identified by these gaugings; survey it or set it "
                f"manually"
            )
        elif h0_diag.get("method") in ("three-point", "fallback"):
            warnings_out.append(
                f"h0 = {fit['h0']:.3f} m came from the fallback ({h0_diag['method']}) "
                f"estimate — curvature was inconclusive; treat h0 as approximate"
            )

    # Bayesian backend: flag a breakpoint the posterior cannot pin down.
    bp_ci = fit.get("breakpoint_ci")
    if bp_ci and fit.get("stage_max") is not None and fit.get("stage_min") is not None:
        span = float(fit["stage_max"]) - float(fit["stage_min"])
        pct = int(round(fit.get("bands", {}).get("level", 0.95) * 100))
        if span > 0:
            for bp, (lo, hi) in zip(fit.get("breakpoints", []), bp_ci):
                if (hi - lo) > 0.33 * span:
                    warnings_out.append(
                        f"breakpoint at {bp:.3f} m is weakly identified — its {pct}% "
                        f"credible interval [{lo:.3f}, {hi:.3f}] spans "
                        f"{100 * (hi - lo) / span:.0f}% of the gauged range; the "
                        f"segment split is uncertain"
                    )

    return warnings_out, critical


def predict_discharge(fit: dict, stage) -> np.ndarray:
    """Evaluate a fitted rating curve (least-squares single/piecewise or
    Bayesian) at given stages."""
    stage = np.atleast_1d(np.asarray(stage, dtype=float))

    if "bayes_params" in fit:
        from rating_curve_automater.bayesian import evaluate_equation

        return evaluate_equation(fit["bayes_params"], stage)

    if not fit.get("is_segmented"):
        return fit["a"] * np.power(np.maximum(stage - fit["h0"], 1e-9), fit["b"])

    return evaluate_spline(fit["spline"], stage)


def _overall_r_squared(
    fit: dict,
    stage: np.ndarray,
    discharge: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    predicted = predict_discharge(fit, stage)
    pw = np.ones_like(discharge) if weights is None else np.asarray(weights, dtype=float) ** 2
    q_bar = float(np.sum(pw * discharge) / np.sum(pw))
    ss_res = float(np.sum(pw * (discharge - predicted) ** 2))
    ss_tot = float(np.sum(pw * (discharge - q_bar) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot != 0 else 1.0


def _fit_single(
    stage: np.ndarray,
    discharge: np.ndarray,
    h0: float,
    h0_estimated: bool,
    weights: np.ndarray | None = None,
) -> dict:
    fit = _loglog_fit(stage, discharge, h0, weights)
    if fit is None:
        raise ValueError(
            f"Not enough points with stage above h0={h0:.3f} to fit a rating curve."
        )
    a, b = fit["a"], fit["b"]
    return {
        "is_segmented": False,
        "a": a,
        "b": b,
        "h0": float(h0),
        "h0_estimated": h0_estimated,
        "r_squared": fit["r_squared"],
        "r_squared_weighted": fit["r_squared_weighted"],
        "n_points": fit["n_points"],
        "n_segments": 1,
        "equation": f"Q = {a:.6f} * (H - {h0:.3f})^{b:.6f}",
    }


def _segment_equation(seg: dict, h0: float, lo: float, hi: float | None) -> str:
    rng = f"H < {hi:.3f}" if hi is not None and lo is None else (
        f"H >= {lo:.3f}" if hi is None else f"{lo:.3f} <= H < {hi:.3f}"
    )
    return f"{rng}: Q = {seg['a']:.6f} * (H - {h0:.3f})^{seg['b']:.6f}"


def _fit_piecewise(
    stage: np.ndarray,
    discharge: np.ndarray,
    h0: float,
    h0_estimated: bool,
    weights: np.ndarray | None = None,
    *,
    n_segments: int | None = None,
    max_segments: int = DEFAULT_MAX_SEGMENTS,
    criterion: str = "bic",
) -> dict:
    """Fit a continuous piecewise power law (see :mod:`rating_curve_automater.piecewise`).

    ``n_segments`` fixes the count; ``None`` lets ``criterion`` choose it (and may
    return a plain single power law when no breakpoint is justified).
    """
    try:
        model = fit_piecewise_power_law(
            stage, discharge, float(h0), weights,
            n_segments=n_segments, max_segments=max_segments, criterion=criterion,
        )
    except np.linalg.LinAlgError as exc:  # pragma: no cover - degenerate design
        raise ValueError(f"Could not fit a piecewise rating curve: {exc}") from exc

    order = np.argsort(stage)
    s_sorted, q_sorted = stage[order], discharge[order]
    w_sorted = weights[order] if weights is not None else None

    if not model["is_segmented"]:
        a, b = model["a"], model["b"]
        fit = {
            "is_segmented": False,
            "a": a,
            "b": b,
            "h0": float(h0),
            "h0_estimated": h0_estimated,
            "n_points": int(np.sum(stage - h0 > 0)),
            "n_segments": 1,
            "segment_selection": "auto" if n_segments is None else "forced",
            "equation": f"Q = {a:.6f} * (H - {h0:.3f})^{b:.6f}",
        }
    else:
        breakpoints = model["breakpoints"]
        parts = []
        for i, seg in enumerate(model["segments"]):
            lo = breakpoints[i - 1] if i > 0 else None
            hi = breakpoints[i] if i < len(breakpoints) else None
            parts.append(_segment_equation(seg, float(h0), lo, hi))
        fit = {
            "is_segmented": True,
            "spline": model["spline"],
            "h0": float(h0),
            "h0_estimated": h0_estimated,
            "breakpoints": breakpoints,
            "breakpoint": breakpoints[0],  # back-compat
            "segments": model["segments"],
            "n_segments": model["n_segments"],
            "n_points": int(np.sum(stage - h0 > 0)),
            "segment_selection": "auto" if n_segments is None else "forced",
            "criterion": model["criterion"],
            "a": model["a"],
            "b": model["b"],
            "equation": "; ".join(parts),
        }

    fit["r_squared"] = _overall_r_squared(fit, s_sorted, q_sorted)
    fit["r_squared_weighted"] = _overall_r_squared(fit, s_sorted, q_sorted, w_sorted)
    return fit


def resolve_point_uncertainty(
    working: pd.DataFrame,
    default_pct: float,
) -> tuple[np.ndarray, str]:
    """Return ``(fractional_uncertainty_per_point, source)``.

    ``source`` is ``"column"`` when a ``discharge_uncertainty`` column supplied
    at least one value (rows without one fall back to ``default_pct``), else
    ``"default"`` (every point gets ``default_pct``).
    """
    n = len(working)
    default_frac = max(float(default_pct), 1e-6) / 100.0

    if DISCHARGE_UNCERTAINTY in working.columns:
        pct = pd.to_numeric(working[DISCHARGE_UNCERTAINTY], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(pct) & (pct > 0)
        if finite.any():
            frac = np.where(finite, pct / 100.0, default_frac)
            # Guard against a column that slipped through as fractions.
            if np.nanmedian(frac[finite]) > 1.0:
                frac = np.where(finite, pct / 10000.0, default_frac)
            return frac, "column"

    return np.full(n, default_frac), "default"


def fit_rating_curve(
    df: pd.DataFrame,
    h0: float | None = None,
    estimate_h0_if_missing: bool = True,
    segments: int | str = 1,
    strict: bool = False,
    discharge_uncertainty_pct: float = DEFAULT_DISCHARGE_UNCERTAINTY_PCT,
    n_bootstrap: int = 0,
    ci_level: float = 0.95,
    random_state: int | None = None,
    max_segments: int = DEFAULT_MAX_SEGMENTS,
    segment_criterion: str = "bic",
    method: str = "ols",
    bayesian_sampler: str = "auto",
    _diagnostics: bool = True,
) -> dict:
    """Fit a power-law rating curve: ``Q = a * (H - h0)^b``.

    The model is fitted on the valid measurement rows in ``df``. When ``h0``
    is not supplied it is estimated from the data (unless
    ``estimate_h0_if_missing`` is ``False``, in which case ``DEFAULT_H0`` is
    used).

    ``segments`` controls the piecewise (multi-control) model:

    * ``1`` – a single power law.
    * an integer ``>= 2`` – a **continuous** piecewise power law with exactly
      that many segments; breakpoint stages are chosen by forward selection.
    * ``"auto"`` – try 1..``max_segments`` and keep the count that minimises
      ``segment_criterion`` (``"bic"`` or ``"aic"``), so the curve only gains a
      segment it can justify.

    ``h0`` is shared across segments. A segmented fit carries ``is_segmented``,
    ``breakpoints`` (list; ``breakpoint`` = the first, for back-compat), a
    ``segments`` list and ``n_segments``; :func:`predict_discharge` evaluates
    any kind.

    ``discharge_uncertainty_pct`` is the assumed measurement uncertainty (in
    percent of Q) for gaugings with no value in a ``discharge_uncertainty``
    column. When such a column is present and its values vary, the fit becomes a
    weighted least-squares regression (noisier gaugings pull the curve less); a
    uniform uncertainty leaves the fit identical to plain OLS.

    ``n_bootstrap`` > 0 adds ``fit["bands"]`` – wild-residual-bootstrap
    confidence and prediction bands over the observed stage range plus
    ``ci_level`` intervals on ``a`` and ``b`` (see
    :mod:`rating_curve_automater.rating_curve_uncertainty`). ``random_state`` makes it repeatable.

    When the gaugings carry dates, ``fit["drift"]`` holds a temporal-drift
    assessment (residual-vs-date frame, time trend, recent-window bias and a
    ``flag``) – see :mod:`rating_curve_automater.rating_curve_drift`.

    ``method="bayesian"`` fits with thodson-usgs ``ratingcurve`` (PyMC) instead
    of least squares; ``h0`` and the bands then come from the posterior and the
    ``[bayesian]`` extra must be installed. ``bayesian_sampler`` picks the
    sampler -- ``"auto"`` (NUTS for small records, ADVI above), ``"nuts"`` or
    ``"advi"``. ``method="ols"`` (default) is the log-log least-squares path
    described above.
    """
    if method not in ("ols", "bayesian"):
        raise ValueError("method must be 'ols' or 'bayesian'.")
    auto_segments = isinstance(segments, str) and segments.lower() == "auto"
    if not auto_segments and (not isinstance(segments, int) or isinstance(segments, bool) or segments < 1):
        raise ValueError("segments must be a positive integer or 'auto'.")

    working = select_valid_measurements(df)
    stage = working[STAGE_COL].to_numpy(dtype=float)
    discharge = working[DISCHARGE_COL].to_numpy(dtype=float)

    if len(stage) < 2:
        raise ValueError("Not enough valid stage-discharge points to fit a rating curve.")

    unc_frac, unc_source = resolve_point_uncertainty(working, discharge_uncertainty_pct)
    weights = 1.0 / np.clip(unc_frac, 1e-6, None)
    is_weighted = bool(unc_source == "column" and not np.allclose(unc_frac, unc_frac[0]))

    if method == "bayesian":
        from rating_curve_automater.bayesian import fit_bayesian_rating_curve

        fit = fit_bayesian_rating_curve(
            stage, discharge, unc_frac,
            segments=segments, level=ci_level, random_state=random_state,
            sampler=bayesian_sampler,
        )
    else:
        h0_estimated = False
        h0_diag: dict | None = None
        if h0 is None:
            if estimate_h0_if_missing:
                h0, h0_diag = _estimate_h0(stage, discharge, weights=weights)
                h0_estimated = True
            else:
                h0 = DEFAULT_H0

        if auto_segments:
            fit = _fit_piecewise(
                stage, discharge, float(h0), h0_estimated, weights,
                n_segments=None, max_segments=max_segments, criterion=segment_criterion,
            )
        elif segments == 1:
            fit = _fit_single(stage, discharge, float(h0), h0_estimated, weights)
        else:
            fit = _fit_piecewise(
                stage, discharge, float(h0), h0_estimated, weights,
                n_segments=int(segments), max_segments=max_segments, criterion=segment_criterion,
            )
        fit["method"] = "ols"
        if h0_diag is not None:
            fit["h0_diagnostics"] = h0_diag

    fit["uncertainty_source"] = unc_source
    fit["uncertainty_pct_default"] = float(discharge_uncertainty_pct)
    fit["mean_uncertainty_pct"] = float(np.mean(unc_frac) * 100.0)
    fit["weighted"] = is_weighted

    used = (stage - float(fit["h0"]) > 0) & (discharge > 0)
    if used.any():
        fit["stage_min"] = float(stage[used].min())
        fit["stage_max"] = float(stage[used].max())

    if method != "bayesian" and n_bootstrap and n_bootstrap > 0:
        from rating_curve_automater.rating_curve_uncertainty import bootstrap_rating_curve

        fit["bands"] = bootstrap_rating_curve(
            stage,
            discharge,
            fit,
            weights=weights,
            n_bootstrap=n_bootstrap,
            level=ci_level,
            random_state=random_state,
            reestimate_h0=bool(fit.get("h0_estimated")),
        )

    if not _diagnostics:
        return fit

    from rating_curve_automater.rating_curve_drift import assess_temporal_drift

    drift = assess_temporal_drift(working, fit, random_state=random_state)
    if drift is not None:
        fit["drift"] = drift

    fit["warnings"], critical = assess_fit(fit, stage, discharge)
    fit["is_plausible"] = not critical
    if strict and critical:
        raise ImplausibleRatingCurve(fit["warnings"])
    return fit


#: Below this many valid gaugings a leave-one-out estimate is too noisy to mean much.
MIN_POINTS_FOR_LOO = 8


def leave_one_out_error(
    df: pd.DataFrame,
    *,
    segments: int | str = 1,
    h0: float | None = None,
    discharge_uncertainty_pct: float = DEFAULT_DISCHARGE_UNCERTAINTY_PCT,
    max_segments: int = DEFAULT_MAX_SEGMENTS,
    segment_criterion: str = "bic",
) -> dict | None:
    """Leave-one-out cross-validated prediction error of the rating curve.

    Each valid gauging is held out in turn, the curve is re-fitted (same
    settings, ``h0`` re-estimated per fold when not supplied) and its discharge
    predicted from the held-out stage. Reports the spread of those out-of-sample
    percentage errors -- an honest accuracy figure, unlike the in-sample R².
    Returns ``None`` below :data:`MIN_POINTS_FOR_LOO` gaugings.
    """
    working = select_valid_measurements(df).reset_index(drop=True)
    n = len(working)
    if n < MIN_POINTS_FOR_LOO:
        return None

    stage = working[STAGE_COL].to_numpy(dtype=float)
    discharge = working[DISCHARGE_COL].to_numpy(dtype=float)
    rel_errors: list[float] = []
    for i in range(n):
        train = working.drop(index=i)
        try:
            f = fit_rating_curve(
                train, h0=h0, segments=segments,
                discharge_uncertainty_pct=discharge_uncertainty_pct,
                max_segments=max_segments, segment_criterion=segment_criterion,
                n_bootstrap=0, _diagnostics=False,
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        pred = float(predict_discharge(f, stage[i])[0])
        if np.isfinite(pred) and discharge[i] > 0:
            rel_errors.append((pred - discharge[i]) / discharge[i])

    if len(rel_errors) < MIN_POINTS_FOR_LOO:
        return None
    err = np.asarray(rel_errors)
    return {
        "n": int(err.size),
        "rmspe_pct": float(100.0 * np.sqrt(np.mean(err ** 2))),
        "bias_pct": float(100.0 * np.mean(err)),
        "mae_pct": float(100.0 * np.mean(np.abs(err))),
        "p95_abs_pct": float(100.0 * np.percentile(np.abs(err), 95)),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fit a rating curve from a cleaned measurements CSV.")
    parser.add_argument("--csv", type=str, default=None, help="Cleaned measurements CSV (default: ./cleaned_measurements.csv).")
    parser.add_argument("--h0", type=float, default=None, help="Stage of zero flow (default: estimate from data).")
    parser.add_argument(
        "--segments", default="1",
        help="1 = single power law; an integer >= 2 = that many continuous segments; 'auto' = pick by BIC.",
    )
    parser.add_argument("--site", type=str, default=None, help="Fit only rows with this value in the 'site' column.")
    parser.add_argument("--method", choices=("ols", "bayesian"), default="ols",
                        help="'ols' = log-log least squares (default); 'bayesian' = ratingcurve/PyMC (needs the [bayesian] extra).")
    parser.add_argument("--sampler", choices=("auto", "nuts", "advi"), default="auto",
                        help="Bayesian sampler: 'auto' (NUTS <=200 gaugings, else ADVI), 'nuts' (exact), 'advi' (fast). Ignored for --method ols.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if the fit is not a plausible rating curve.")
    parser.add_argument(
        "--uncertainty-pct",
        type=float,
        default=DEFAULT_DISCHARGE_UNCERTAINTY_PCT,
        help=(
            "Assumed discharge measurement uncertainty in percent for points with no "
            f"'discharge_uncertainty' value (default: {DEFAULT_DISCHARGE_UNCERTAINTY_PCT})."
        ),
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=DEFAULT_N_BOOTSTRAP,
        help=f"Bootstrap replicates for the confidence/prediction bands (0 = off; default: {DEFAULT_N_BOOTSTRAP}).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the bootstrap.")
    parser.add_argument("--loo", action="store_true",
                        help="Also report leave-one-out cross-validated prediction error (honest out-of-sample accuracy).")
    parser.add_argument("--cross-section", type=str, default=None,
                        help="Cross-section CSV (offset + elevation columns) for a Manning sanity check of the curve's extrapolation.")
    parser.add_argument("--slope", type=float, default=None,
                        help="Channel/energy slope (m/m) — required with --cross-section.")
    parser.add_argument("--mannings-n", type=float, default=None,
                        help="Manning's n for the check (default: calibrate it to the rating over the gauged range).")
    parser.add_argument("--stage-offset", type=float, default=0.0,
                        help="Add this to stage H to get water-surface elevation in the cross-section's datum (default 0).")
    args = parser.parse_args()
    if args.cross_section and args.slope is None:
        raise SystemExit("--cross-section needs --slope (channel slope in m/m).")

    csv_path = Path(args.csv) if args.csv else Path("cleaned_measurements.csv")
    df = pd.read_csv(csv_path)
    if args.site is not None:
        from rating_curve_automater.schema import SITE

        if SITE not in df.columns:
            raise SystemExit("No 'site' column in the CSV.")
        df = df[df[SITE].astype(str).str.strip() == args.site]
        print(f"Site filter: {args.site} ({len(df)} rows)")
    segments: int | str = args.segments if args.segments.lower() == "auto" else int(args.segments)
    try:
        fit = fit_rating_curve(
            df,
            h0=args.h0,
            segments=segments,
            strict=args.strict,
            discharge_uncertainty_pct=args.uncertainty_pct,
            n_bootstrap=args.bootstrap,
            random_state=args.seed,
            method=args.method,
            bayesian_sampler=args.sampler,
        )
    except ImplausibleRatingCurve as exc:
        raise SystemExit(f"Implausible rating curve:\n  - " + "\n  - ".join(exc.warnings))
    except ImportError as exc:
        raise SystemExit(str(exc))

    print("Rating curve fit results")
    if fit.get("method") == "bayesian":
        print(f"estimator = Bayesian (ratingcurve/PyMC, {fit.get('bayes', {}).get('sampler', '?')})")
    else:
        print("estimator = log-log least squares")
    h0_diag = fit.get("h0_diagnostics")
    if fit["h0_estimated"] and h0_diag is not None:
        h0_note = f"estimated, {h0_diag['method']}" + (", weakly identified" if h0_diag.get("railed") else "")
    else:
        h0_note = "estimated" if fit["h0_estimated"] else "fixed"
    print(f"h0 = {fit['h0']:.3f} ({h0_note})")
    if fit["uncertainty_source"] == "column":
        kind = "weighted least squares" if fit["weighted"] else "uniform (unweighted)"
        print(f"discharge uncertainty = per-point column, {kind}")
    else:
        print(f"discharge uncertainty = {fit['uncertainty_pct_default']:.1f}% assumed (unweighted)")
    if fit["is_segmented"]:
        pick = " (chosen by " + fit.get("criterion", "bic").upper() + ")" if fit.get("segment_selection") == "auto" else ""
        print(f"{fit['n_segments']} segments{pick}; breakpoints at "
              + ", ".join(f"{bp:.3f}" for bp in fit["breakpoints"]) + " m")
        for i, seg in enumerate(fit["segments"], start=1):
            print(
                f"segment {i} [{seg['stage_min']:.3f}-{seg['stage_max']:.3f} m]: "
                f"a={seg['a']:.6f} b={seg['b']:.6f} R²={seg['r_squared']:.4f} n={seg['n_points']}"
            )
    else:
        print(f"a = {fit['a']:.6f}")
        print(f"b = {fit['b']:.6f}")
    print(f"Overall R^2 = {fit['r_squared']:.4f}")
    if fit.get("weighted"):
        print(f"Weighted R^2 = {fit['r_squared_weighted']:.4f}")
    print(f"points = {fit['n_points']}")
    print(f"Equation: {fit['equation']}")
    if args.loo:
        seg_arg: int | str = "auto" if fit.get("segment_selection") == "auto" else (
            fit.get("n_segments", 1) if fit.get("n_segments", 1) > 1 else 1)
        loo = leave_one_out_error(
            df, segments=seg_arg, h0=args.h0,
            discharge_uncertainty_pct=args.uncertainty_pct,
        )
        if loo:
            print(
                f"Leave-one-out: RMSPE {loo['rmspe_pct']:.1f}%  bias {loo['bias_pct']:+.1f}%  "
                f"MAE {loo['mae_pct']:.1f}%  (n={loo['n']}, honest out-of-sample)"
            )
    if args.cross_section:
        from rating_curve_automater.manning import manning_sanity_check, read_cross_section

        offset, bed = read_cross_section(args.cross_section)
        mc = manning_sanity_check(
            fit, offset, bed, args.slope,
            n=args.mannings_n, stage_offset=args.stage_offset,
        )
        print(f"\nManning cross-section check [{mc['flag']}]: {mc['message']}")
        if np.isfinite(mc.get("max_abs_pct_diff_extrapolated", float("nan"))):
            print(
                f"  n {'supplied' if mc.get('n_supplied') else 'calibrated'} = "
                f"{mc['n_used']:.3f}; fitted vs Manning differs up to "
                f"{mc['max_abs_pct_diff_gauged']:.0f}% in-range, "
                f"{mc['max_abs_pct_diff_extrapolated']:.0f}% extrapolated "
                f"(to {mc['extrapolation_ceiling']:.2f} m)"
            )
    bands = fit.get("bands")
    if bands:
        pct = int(round(bands["level"] * 100))
        if bands.get("b_ci"):
            lo, hi = bands["b_ci"]
            print(f"{pct}% CI: b in [{lo:.3f}, {hi:.3f}]")
        if bands.get("a_ci"):
            lo, hi = bands["a_ci"]
            print(f"{pct}% CI: a in [{lo:.4f}, {hi:.4f}]")
        if bands.get("h0_ci"):
            lo, hi = bands["h0_ci"]
            print(f"{pct}% CI: h0 in [{lo:.3f}, {hi:.3f}] (propagated through the band)")
        unit = "posterior draws" if bands.get("kind") == "posterior" else "bootstrap replicates"
        print(
            f"{pct}% confidence band half-width at median stage: "
            f"±{bands['ci_halfwidth_pct_at_median']:.1f}%  "
            f"({bands['n_success']} {unit})"
        )
    drift = fit.get("drift")
    if drift:
        print(f"\nTemporal drift [{drift['flag']}]: {drift['message']}")
        if drift["flag"] != "none":
            print(
                f"  trend {drift.get('trend_pct_per_year', float('nan')):+.1f}%/yr "
                f"(p={drift.get('trend_p_value', float('nan')):.3f}); "
                f"recent {drift.get('recent_n', 0)} gaugings {drift.get('recent_mean_pct', float('nan')):+.0f}% "
                f"(p={drift.get('recent_p_value', float('nan')):.3f})"
            )
    if fit["warnings"]:
        marker = "NOT A PLAUSIBLE RATING CURVE" if not fit["is_plausible"] else "warnings"
        print(f"\n[{marker}]")
        for warning in fit["warnings"]:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
