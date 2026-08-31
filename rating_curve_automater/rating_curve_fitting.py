from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rating_curve_automater.schema import DISCHARGE_CMS, STAGE_M, ensure_canonical

DEFAULT_H0 = 0.18

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


def _loglog_fit(stage: np.ndarray, discharge: np.ndarray, h0: float) -> dict | None:
    """Fit ``Q = a * (H - h0)^b`` by linear regression in log-log space.

    Returns ``None`` when fewer than two points have ``H - h0 > 0``.
    """
    x = stage - h0
    mask = x > 0
    if int(mask.sum()) < 2:
        return None

    discharge_m = discharge[mask]
    x_m = x[mask]

    slope, intercept = np.polyfit(np.log(x_m), np.log(discharge_m), 1)
    a = float(np.exp(intercept))
    b = float(slope)

    predicted = a * np.power(x_m, b)
    residuals = discharge_m - predicted
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((discharge_m - np.mean(discharge_m)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot != 0 else 1.0

    return {
        "a": a,
        "b": b,
        "r_squared": float(r_squared),
        "ss_res": ss_res,
        "n_points": int(mask.sum()),
    }


def estimate_h0(
    stage: np.ndarray,
    discharge: np.ndarray,
    bounds: tuple[float, float] | None = None,
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
        fit = _loglog_fit(stage, discharge, h0)
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

    if fit["r_squared"] < MIN_R_SQUARED:
        warnings_out.append(f"poor fit: R² = {fit['r_squared']:.2f}")
    if fit["n_points"] < MIN_POINTS_RELIABLE:
        warnings_out.append(f"only {fit['n_points']} point(s) — the fit is unreliable")

    return warnings_out, critical


def predict_discharge(fit: dict, stage) -> np.ndarray:
    """Evaluate a fitted rating curve (single or segmented) at given stages."""
    stage = np.atleast_1d(np.asarray(stage, dtype=float))
    h0 = fit["h0"]

    if not fit.get("is_segmented"):
        return fit["a"] * np.power(np.maximum(stage - h0, 1e-9), fit["b"])

    breakpoint = fit["breakpoint"]
    lower, upper = fit["segments"]
    x = np.maximum(stage - h0, 1e-9)
    below = stage < breakpoint
    out = np.where(
        below,
        lower["a"] * np.power(x, lower["b"]),
        upper["a"] * np.power(x, upper["b"]),
    )
    return out


def _overall_r_squared(fit: dict, stage: np.ndarray, discharge: np.ndarray) -> float:
    predicted = predict_discharge(fit, stage)
    ss_res = float(np.sum((discharge - predicted) ** 2))
    ss_tot = float(np.sum((discharge - np.mean(discharge)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot != 0 else 1.0


def _fit_single(
    stage: np.ndarray,
    discharge: np.ndarray,
    h0: float,
    h0_estimated: bool,
) -> dict:
    fit = _loglog_fit(stage, discharge, h0)
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
        "n_points": fit["n_points"],
        "equation": f"Q = {a:.6f} * (H - {h0:.3f})^{b:.6f}",
    }


def _fit_two_segments(
    stage: np.ndarray,
    discharge: np.ndarray,
    h0: float,
    h0_estimated: bool,
) -> dict:
    order = np.argsort(stage)
    stage, discharge = stage[order], discharge[order]

    min_points = max(MIN_SEGMENT_POINTS, int(0.15 * len(stage)))
    unique_stages = np.unique(stage)
    best = None
    for breakpoint in unique_stages:
        below = stage < breakpoint
        if int(below.sum()) < min_points or int((~below).sum()) < min_points:
            continue
        lower = _loglog_fit(stage[below], discharge[below], h0)
        upper = _loglog_fit(stage[~below], discharge[~below], h0)
        if lower is None or upper is None:
            continue
        ss_res = 0.0
        for seg_fit, seg_mask in ((lower, below), (upper, ~below)):
            predicted = seg_fit["a"] * np.power(np.maximum(stage[seg_mask] - h0, 1e-9), seg_fit["b"])
            ss_res += float(np.sum((discharge[seg_mask] - predicted) ** 2))
        if best is None or ss_res < best[0]:
            best = (ss_res, float(breakpoint), lower, upper)

    if best is None:
        raise ValueError(
            "Not enough points on both sides of any breakpoint to fit a segmented curve; "
            "use segments=1."
        )

    _, breakpoint, lower, upper = best

    def segment_record(seg_fit: dict, seg_stage: np.ndarray) -> dict:
        return {
            "a": seg_fit["a"],
            "b": seg_fit["b"],
            "r_squared": seg_fit["r_squared"],
            "n_points": seg_fit["n_points"],
            "stage_min": float(seg_stage.min()),
            "stage_max": float(seg_stage.max()),
        }

    below = stage < breakpoint
    fit = {
        "is_segmented": True,
        "h0": float(h0),
        "h0_estimated": h0_estimated,
        "breakpoint": breakpoint,
        "segments": [segment_record(lower, stage[below]), segment_record(upper, stage[~below])],
        "n_points": int(len(stage)),
        # back-compat scalars describe the lower segment
        "a": lower["a"],
        "b": lower["b"],
        "equation": (
            f"H < {breakpoint:.3f}: Q = {lower['a']:.6f} * (H - {h0:.3f})^{lower['b']:.6f}; "
            f"H >= {breakpoint:.3f}: Q = {upper['a']:.6f} * (H - {h0:.3f})^{upper['b']:.6f}"
        ),
    }
    fit["r_squared"] = _overall_r_squared(fit, stage, discharge)
    return fit


def fit_rating_curve(
    df: pd.DataFrame,
    h0: float | None = None,
    estimate_h0_if_missing: bool = True,
    segments: int = 1,
    strict: bool = False,
) -> dict:
    """Fit a power-law rating curve: ``Q = a * (H - h0)^b``.

    The model is fitted on the valid measurement rows in ``df``. When ``h0``
    is not supplied it is estimated from the data (unless
    ``estimate_h0_if_missing`` is ``False``, in which case ``DEFAULT_H0`` is
    used).

    ``segments=2`` fits a piecewise curve, searching for the breakpoint stage
    that minimises the combined residual sum of squares (useful when a site
    behaves differently at low vs. high flow). ``h0`` is shared across
    segments. The returned dict then carries ``is_segmented=True``,
    ``breakpoint`` and a ``segments`` list; :func:`predict_discharge` evaluates
    either kind.
    """
    if segments not in (1, 2):
        raise ValueError("segments must be 1 or 2.")

    working = select_valid_measurements(df)
    stage = working[STAGE_COL].to_numpy(dtype=float)
    discharge = working[DISCHARGE_COL].to_numpy(dtype=float)

    if len(stage) < 2:
        raise ValueError("Not enough valid stage-discharge points to fit a rating curve.")

    h0_estimated = False
    if h0 is None:
        if estimate_h0_if_missing:
            h0 = estimate_h0(stage, discharge)
            h0_estimated = True
        else:
            h0 = DEFAULT_H0

    if segments == 1:
        fit = _fit_single(stage, discharge, float(h0), h0_estimated)
    else:
        fit = _fit_two_segments(stage, discharge, float(h0), h0_estimated)

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
    parser.add_argument("--segments", type=int, default=1, choices=(1, 2), help="1 = single power law, 2 = piecewise.")
    parser.add_argument("--site", type=str, default=None, help="Fit only rows with this value in the 'site' column.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if the fit is not a plausible rating curve.")
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else Path(__file__).resolve().parent.parent / "cleaned_measurements.csv"
    df = pd.read_csv(csv_path)
    if args.site is not None:
        from rating_curve_automater.schema import SITE

        if SITE not in df.columns:
            raise SystemExit("No 'site' column in the CSV.")
        df = df[df[SITE].astype(str).str.strip() == args.site]
        print(f"Site filter: {args.site} ({len(df)} rows)")
    try:
        fit = fit_rating_curve(df, h0=args.h0, segments=args.segments, strict=args.strict)
    except ImplausibleRatingCurve as exc:
        raise SystemExit(f"Implausible rating curve:\n  - " + "\n  - ".join(exc.warnings))

    print("Rating curve fit results")
    print(f"h0 = {fit['h0']:.3f} ({'estimated' if fit['h0_estimated'] else 'fixed'})")
    if fit["is_segmented"]:
        print(f"breakpoint stage = {fit['breakpoint']:.3f} m")
        for i, seg in enumerate(fit["segments"], start=1):
            print(
                f"segment {i} [{seg['stage_min']:.3f}-{seg['stage_max']:.3f} m]: "
                f"a={seg['a']:.6f} b={seg['b']:.6f} R²={seg['r_squared']:.4f} n={seg['n_points']}"
            )
    else:
        print(f"a = {fit['a']:.6f}")
        print(f"b = {fit['b']:.6f}")
    print(f"Overall R^2 = {fit['r_squared']:.4f}")
    print(f"points = {fit['n_points']}")
    print(f"Equation: {fit['equation']}")
    if fit["warnings"]:
        marker = "NOT A PLAUSIBLE RATING CURVE" if not fit["is_plausible"] else "warnings"
        print(f"\n[{marker}]")
        for warning in fit["warnings"]:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
