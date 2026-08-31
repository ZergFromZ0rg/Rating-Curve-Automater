"""Bootstrap confidence and prediction bands for a fitted rating curve.

The point fit (:func:`rating_curve_automater.rating_curve_fitting.fit_rating_curve`) gives a single
``Q = a·(H − h0)^b`` curve. This module resamples the fit to answer two further
questions:

* **Confidence band** – how well is the *mean* stage–discharge relationship
  pinned down by the gaugings we have? (narrow)
* **Prediction band** – where would the *next* gauging at a given stage most
  likely fall? (wider – it also carries the scatter of individual gaugings)

Method: a wild residual bootstrap in log space. The fitted log-residuals are
re-signed with standard-normal multipliers and the curve is re-fitted with the
same measurement-uncertainty weights; percentiles are taken across the replicates
on a dense stage grid. When ``h0`` was estimated from the data it is
**re-estimated inside every replicate** so the band carries the uncertainty in
the point of zero flow (holding it fixed makes the low-flow band too tight);
the physical breakpoint stages of a segmented curve are held at their point
estimates. The band spans the observed stage range only – it is not an
extrapolation tool.
"""

from __future__ import annotations

import numpy as np

DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_CI_LEVEL = 0.95
_GRID_POINTS = 100
#: Below this many usable gaugings a bootstrap band is too unstable to report.
MIN_POINTS_FOR_BANDS = 4


def _refit_fixed(
    stage_m: np.ndarray,
    discharge_star: np.ndarray,
    fit: dict,
    weights_m: np.ndarray | None,
    h0: float,
) -> dict | None:
    """Re-fit on resampled discharge at the given ``h0``, physical breakpoints held.

    ``h0`` is passed in (it may be re-estimated per replicate). Points that fall
    at or below it are dropped for this replicate. Returns a minimal fit dict
    understood by :func:`rating_curve_automater.rating_curve_fitting.predict_discharge`,
    or ``None`` if it could not be fitted.
    """
    from rating_curve_automater.rating_curve_fitting import _loglog_fit

    usable = stage_m - h0 > 1e-9
    if int(usable.sum()) < MIN_POINTS_FOR_BANDS:
        return None
    s, q = stage_m[usable], discharge_star[usable]
    w = None if weights_m is None else weights_m[usable]

    if not fit.get("is_segmented"):
        r = _loglog_fit(s, q, h0, w)
        if r is None:
            return None
        return {"is_segmented": False, "a": r["a"], "b": r["b"], "h0": h0}

    # Piecewise: keep the physical breakpoint stages, re-derive the spline knots
    # in log space from this replicate's h0, re-fit the spline coefficients.
    from rating_curve_automater.piecewise import fit_spline_coef

    breakpoints = fit.get("breakpoints") or [fit["breakpoint"]]
    if any(bp - h0 <= 1e-9 for bp in breakpoints):
        return None
    knots_u = [float(np.log(bp - h0)) for bp in breakpoints]
    u = np.log(s - h0)
    y = np.log(np.maximum(q, 1e-12))
    try:
        coef = fit_spline_coef(u, y, w, knots_u)
    except np.linalg.LinAlgError:
        return None
    return {
        "is_segmented": True,
        "spline": {"h0": h0, "knots_u": knots_u, "coef": [float(c) for c in coef]},
    }


def bootstrap_rating_curve(
    stage: np.ndarray,
    discharge: np.ndarray,
    fit: dict,
    weights: np.ndarray | None = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    level: float = DEFAULT_CI_LEVEL,
    random_state: int | None = None,
    reestimate_h0: bool = False,
) -> dict | None:
    """Return confidence/prediction bands for ``fit`` over the observed stages.

    ``stage`` / ``discharge`` are the valid gaugings the curve was fitted on and
    ``weights`` the per-point regression weights (``1 / fractional uncertainty``).
    ``reestimate_h0`` re-estimates the point of zero flow inside each replicate
    (pass the fit's ``h0_estimated``) so the band reflects that uncertainty too.
    Returns ``None`` when there are too few points for a meaningful band.
    """
    from rating_curve_automater.rating_curve_fitting import _estimate_h0, predict_discharge

    stage = np.asarray(stage, dtype=float)
    discharge = np.asarray(discharge, dtype=float)
    h0 = fit["h0"]

    keep = (stage - h0 > 0) & (discharge > 0)
    stage_m = stage[keep]
    discharge_m = discharge[keep]
    weights_m = np.asarray(weights, dtype=float)[keep] if weights is not None else None
    n = stage_m.size
    if n < MIN_POINTS_FOR_BANDS or n_bootstrap < 100:
        return None

    log_q = np.log(discharge_m)
    log_q_hat = np.log(np.maximum(predict_discharge(fit, stage_m), 1e-12))
    resid = log_q - log_q_hat
    resid_centered = resid - resid.mean()

    grid = np.linspace(stage_m.min(), stage_m.max(), _GRID_POINTS)
    q_grid = predict_discharge(fit, grid)

    rng = np.random.default_rng(random_state)
    curve_samples: list[np.ndarray] = []
    a_samples: list[float] = []
    b_samples: list[float] = []
    h0_samples: list[float] = []

    for _ in range(int(n_bootstrap)):
        multipliers = rng.standard_normal(n)
        discharge_star = np.exp(log_q_hat + resid * multipliers)
        if reestimate_h0:
            h0_star = _estimate_h0(stage_m, discharge_star, weights=weights_m)[0]
        else:
            h0_star = h0
        refit = _refit_fixed(stage_m, discharge_star, fit, weights_m, h0_star)
        if refit is None:
            continue
        curve_samples.append(predict_discharge(refit, grid))
        h0_samples.append(h0_star)
        if not refit["is_segmented"]:
            a_samples.append(refit["a"])
            b_samples.append(refit["b"])

    if len(curve_samples) < 100:
        return None

    curves = np.vstack(curve_samples)                       # (B, G)
    lo_pct = 100.0 * (1.0 - level) / 2.0
    hi_pct = 100.0 * (1.0 + level) / 2.0

    ci_lower, ci_upper = np.percentile(curves, [lo_pct, hi_pct], axis=0)

    # Prediction band: parameter uncertainty (curves) plus the scatter of an
    # individual gauging (a resampled log-residual per grid point per replicate).
    noise = rng.choice(resid_centered, size=curves.shape, replace=True)
    pi = np.exp(np.log(np.maximum(curves, 1e-12)) + noise)
    pi_lower, pi_upper = np.percentile(pi, [lo_pct, hi_pct], axis=0)

    # Keep the bands ordered and never tighter than the curve itself.
    ci_lower = np.minimum(ci_lower, q_grid)
    ci_upper = np.maximum(ci_upper, q_grid)
    pi_lower = np.minimum(pi_lower, ci_lower)
    pi_upper = np.maximum(pi_upper, ci_upper)

    def _pair(values: list[float]) -> tuple[float, float] | None:
        if len(values) < 100:
            return None
        return tuple(float(v) for v in np.percentile(values, [lo_pct, hi_pct]))

    median_stage = float(np.median(stage_m))
    q_at_median = float(np.interp(median_stage, grid, q_grid))
    ci_hw = float(np.interp(median_stage, grid, (ci_upper - ci_lower) / 2.0))
    ci_halfwidth_pct = 100.0 * ci_hw / q_at_median if q_at_median > 0 else float("nan")

    return {
        "kind": "bootstrap",
        "level": float(level),
        "n_bootstrap": int(n_bootstrap),
        "n_success": len(curve_samples),
        "stage": grid,
        "q": q_grid,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "pi_lower": pi_lower,
        "pi_upper": pi_upper,
        "a_ci": _pair(a_samples),
        "b_ci": _pair(b_samples),
        "h0_ci": _pair(h0_samples) if reestimate_h0 else None,
        "h0_reestimated": bool(reestimate_h0),
        "ci_halfwidth_pct_at_median": ci_halfwidth_pct,
    }
