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


def estimate_h0(
    stage: np.ndarray,
    discharge: np.ndarray,
    bounds: tuple[float, float] | None = None,
    weights: np.ndarray | None = None,
) -> float:
    """Estimate the stage of zero flow ``h0`` by maximising the fit R².

    A golden-section search over ``h0`` is used. The search is capped just
    below ``min(stage)`` so every measurement keeps ``H - h0 > 0`` and the
    objective is compared on the same points at every candidate ``h0``.
    """
    stage = np.asarray(stage, dtype=float)
    discharge = np.asarray(discharge, dtype=float)

    stage_min = float(np.min(stage))
    if bounds is None:
        lo, hi = 0.0, 0.95 * stage_min
    else:
        lo, hi = bounds
        lo = max(lo, 0.0)
        hi = min(hi, 0.999 * stage_min)

    if not hi > lo:
        return DEFAULT_H0

    def objective(h0: float) -> float:
        fit = _loglog_fit(stage, discharge, h0, weights)
        return fit["r_squared"] if fit is not None else -np.inf

    golden = (np.sqrt(5.0) - 1.0) / 2.0
    c = hi - golden * (hi - lo)
    d = lo + golden * (hi - lo)
    fc, fd = objective(c), objective(d)

    for _ in range(100):
        if hi - lo < 1e-6:
            break
        if fc > fd:
            hi, d, fd = d, c, fc
            c = hi - golden * (hi - lo)
            fc = objective(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + golden * (hi - lo)
            fd = objective(d)

    return float((lo + hi) / 2.0)


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
    ``[bayesian]`` extra must be installed. ``method="ols"`` (default) is the
    log-log least-squares path described above.
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
        )
    else:
        h0_estimated = False
        if h0 is None:
            if estimate_h0_if_missing:
                h0 = estimate_h0(stage, discharge, weights=weights)
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
        )

    from rating_curve_automater.rating_curve_drift import assess_temporal_drift

    drift = assess_temporal_drift(working, fit, random_state=random_state)
    if drift is not None:
        fit["drift"] = drift

    fit["warnings"], critical = assess_fit(fit, stage, discharge)
    fit["is_plausible"] = not critical
    if strict and critical:
        raise ImplausibleRatingCurve(fit["warnings"])
    return fit


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
    args = parser.parse_args()

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
        )
    except ImplausibleRatingCurve as exc:
        raise SystemExit(f"Implausible rating curve:\n  - " + "\n  - ".join(exc.warnings))
    except ImportError as exc:
        raise SystemExit(str(exc))

    print("Rating curve fit results")
    print(f"estimator = {'Bayesian (ratingcurve/PyMC)' if fit.get('method') == 'bayesian' else 'log-log least squares'}")
    print(f"h0 = {fit['h0']:.3f} ({'estimated' if fit['h0_estimated'] else 'fixed'})")
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
    bands = fit.get("bands")
    if bands:
        pct = int(round(bands["level"] * 100))
        if bands.get("b_ci"):
            lo, hi = bands["b_ci"]
            print(f"{pct}% CI: b in [{lo:.3f}, {hi:.3f}]")
        if bands.get("a_ci"):
            lo, hi = bands["a_ci"]
            print(f"{pct}% CI: a in [{lo:.4f}, {hi:.4f}]")
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
