"""Temporal-drift diagnostics for a fitted rating curve.

A rating curve ages: the channel scours (bed lowers → more discharge at a given
stage) or fills (aggradation, vegetation → less), so gaugings made years apart
no longer sit on one static curve. This module measures that from the residuals
of the fitted curve against gauging date and raises a **shift flag** when recent
gaugings sit systematically off the curve.

``assess_temporal_drift`` returns ``None`` when the gaugings carry no dates or
are too few / too closely spaced in time to say anything; otherwise a dict with
a per-gauging residual frame, a linear time-trend (with a permutation p-value),
a recent-window bias test, a split-period test that re-fits the early and late
halves of the record and compares them (this catches a shift the combined fit
absorbed into curve shape), and a ``flag`` of ``"none"`` / ``"possible"`` /
``"likely"`` / ``"unassessable"`` (the last when stage and date are so
correlated that a shift cannot be separated from the curve's shape).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rating_curve_automater.schema import DATE, DISCHARGE_CMS, STAGE_M

#: Need at least this many dated gaugings spanning at least this many days.
MIN_POINTS = 6
MIN_SPAN_DAYS = 45
_N_PERM = 2000
#: Split-period shift test: gaugings per half, and permutations for its p-value.
_MIN_PER_HALF = 6
_N_PERM_SPLIT = 400

#: A time trend of this many percent per year (or a recent-window mean offset of
#: this many percent) is treated as hydrologically material.
TREND_RATE_PCT = 3.0
RECENT_BIAS_PCT = 7.0
_ALPHA = 0.05
_ALPHA_WEAK = 0.15
_RECENT_WINDOW_DAYS = 365

OUT_DATE = "Date"
OUT_STAGE = "Stage (m)"
OUT_OBS = "Observed Q (m³/s)"
OUT_MOD = "Modelled Q (m³/s)"
OUT_PCT = "Residual (%)"
OUT_LOG = "Log residual"
OUT_SHIFT = "Stage shift (m)"


def build_residual_frame(working: pd.DataFrame, fit: dict) -> pd.DataFrame | None:
    """Per-gauging residuals of ``fit`` against date, sorted by date.

    ``working`` must have canonical ``stage_m`` / ``discharge_cms`` columns and,
    for a non-empty result, a ``date`` column. Returns ``None`` when there is no
    usable date or fewer than :data:`MIN_POINTS` dated gaugings.
    """
    from rating_curve_automater.rating_curve_fitting import predict_discharge

    if DATE not in working.columns:
        return None

    sub = working[[c for c in (DATE, STAGE_M, DISCHARGE_CMS) if c in working.columns]].copy()
    sub[DATE] = pd.to_datetime(sub[DATE], errors="coerce")
    sub = sub.dropna(subset=[DATE, STAGE_M, DISCHARGE_CMS])
    sub = sub[(sub[STAGE_M] > fit["h0"]) & (sub[DISCHARGE_CMS] > 0)]
    if len(sub) < MIN_POINTS:
        return None

    sub = sub.sort_values(DATE).reset_index(drop=True)
    stage = sub[STAGE_M].to_numpy(dtype=float)
    obs = sub[DISCHARGE_CMS].to_numpy(dtype=float)
    modelled = np.asarray(predict_discharge(fit, stage), dtype=float)

    log_resid = np.log(obs / np.maximum(modelled, 1e-12))

    # Stage shift: the rating's stage for the gauging's discharge, minus the
    # gauging's own stage (curve inverted numerically; it is monotone). Positive
    # = the gauging plots below/right of the rating (section conveying more than
    # the rating shows, e.g. scour) — same sign as the discharge residual.
    grid_h = np.linspace(stage.min(), stage.max(), 400)
    grid_h = grid_h[grid_h > fit["h0"]]
    grid_q = np.asarray(predict_discharge(fit, grid_h), dtype=float)
    order = np.argsort(grid_q)
    stage_at_q = np.interp(obs, grid_q[order], grid_h[order])

    out = pd.DataFrame({
        OUT_DATE: sub[DATE].dt.date.astype(str),
        OUT_STAGE: stage,
        OUT_OBS: obs,
        OUT_MOD: modelled,
        OUT_PCT: (np.exp(log_resid) - 1.0) * 100.0,
        OUT_LOG: log_resid,
        OUT_SHIFT: stage_at_q - stage,
    })
    out.attrs["dates"] = sub[DATE]
    return out


def _perm_slope_p(t: np.ndarray, y: np.ndarray, slope_obs: float, rng) -> float:
    tc = t - t.mean()
    denom = float(np.sum(tc ** 2)) or 1e-12
    perms = rng.permuted(np.broadcast_to(y, (_N_PERM, y.size)), axis=1)
    slopes = (perms @ tc) / denom
    return float((np.sum(np.abs(slopes) >= abs(slope_obs) - 1e-15) + 1) / (_N_PERM + 1))


def _perm_group_p(y: np.ndarray, recent_mask: np.ndarray, diff_obs: float, rng) -> float:
    n = y.size
    nr = int(recent_mask.sum())
    ne = n - nr
    if nr == 0 or ne == 0:
        return 1.0
    labels = rng.permuted(np.broadcast_to(recent_mask, (_N_PERM, n)), axis=1).astype(float)
    sum_recent = labels @ y
    md = sum_recent / nr - (y.sum() - sum_recent) / ne
    return float((np.sum(np.abs(md) >= abs(diff_obs) - 1e-15) + 1) / (_N_PERM + 1))


def _loglog_ab(x_pos: np.ndarray, q_pos: np.ndarray) -> tuple[float, float] | None:
    """``(a, b)`` for ``Q = a * x^b`` by log-log least squares. ``x`` is already
    ``H - h0`` and all positive."""
    if x_pos.size < 3 or np.unique(x_pos).size < 2:
        return None
    slope, intercept = np.polyfit(np.log(x_pos), np.log(q_pos), 1)
    return float(np.exp(intercept)), float(slope)


def _median_pct_gap(params_a, params_b, x_lo: float, x_hi: float) -> float:
    """Median percent difference of curve B relative to curve A over ``[x_lo, x_hi]``."""
    xs = np.linspace(x_lo, x_hi, 50)
    qa = params_a[0] * xs ** params_a[1]
    qb = params_b[0] * xs ** params_b[1]
    return float(np.median((qb - qa) / qa) * 100.0)


def _period_split_shift(stage: np.ndarray, obs: np.ndarray, order_by_date: np.ndarray,
                        h0: float, rng) -> dict | None:
    """Fit the low and the high half of the record (by date) separately and
    compare the two curves over their shared stage range.

    Catches a shift that a single combined fit absorbs into curve *shape* --
    which happens whenever the gauging schedule ties stage to season. ``None``
    when either half is too small or the halves barely overlap in stage.
    """
    x = stage - h0
    keep = (x > 0) & (obs > 0)
    x, q, idx = x[keep], obs[keep], order_by_date[keep]
    n = x.size
    if n < 2 * _MIN_PER_HALF:
        return None

    early = idx < n / 2.0            # idx is 0..n-1 in date order
    late = ~early
    if early.sum() < _MIN_PER_HALF or late.sum() < _MIN_PER_HALF:
        return None

    lo = max(x[early].min(), x[late].min())
    hi = min(x[early].max(), x[late].max())
    if not hi > lo * 1.15:           # need a real shared stage range
        return None

    fit_e = _loglog_ab(x[early], q[early])
    fit_l = _loglog_ab(x[late], q[late])
    if fit_e is None or fit_l is None:
        return None
    gap_obs = _median_pct_gap(fit_e, fit_l, lo, hi)

    # Permutation null: shuffle which gaugings are "early" vs "late".
    hits = 1
    for _ in range(_N_PERM_SPLIT):
        perm = rng.permutation(n)
        pe, pl = perm[: n // 2], perm[n // 2:]
        fe = _loglog_ab(x[pe], q[pe])
        fl = _loglog_ab(x[pl], q[pl])
        if fe is None or fl is None:
            continue
        if abs(_median_pct_gap(fe, fl, lo, hi)) >= abs(gap_obs) - 1e-12:
            hits += 1
    return {
        "shift_pct": gap_obs,
        "p_value": hits / (_N_PERM_SPLIT + 1),
        "n_early": int(early.sum()),
        "n_late": int(late.sum()),
    }


def _direction(pct: float) -> str:
    if pct >= 0:
        return "above the curve (observed discharge higher than modelled — possible channel scour / control degradation)"
    return "below the curve (observed discharge lower than modelled — possible aggradation, backwater or vegetation growth)"


def assess_temporal_drift(
    working: pd.DataFrame,
    fit: dict,
    random_state: int | None = None,
) -> dict | None:
    """Detect systematic change in the rating-curve residuals over time."""
    frame = build_residual_frame(working, fit)
    if frame is None:
        return None

    dates: pd.Series = frame.attrs["dates"].reset_index(drop=True)
    span_days = int((dates.iloc[-1] - dates.iloc[0]).days)
    n = len(frame)
    y = frame[OUT_LOG].to_numpy(dtype=float)
    residuals = frame.copy()
    residuals.attrs.clear()
    result: dict = {
        "n": n,
        "date_min": str(dates.iloc[0].date()),
        "date_max": str(dates.iloc[-1].date()),
        "span_days": span_days,
        "residuals": residuals,
        "flag": "none",
    }

    if span_days < MIN_SPAN_DAYS:
        result["message"] = (
            f"Gaugings span only {span_days} days — too short to assess drift."
        )
        return result

    rng = np.random.default_rng(random_state)
    t_years = (dates - dates.iloc[0]).dt.total_seconds().to_numpy() / (365.25 * 86400.0)

    # If stage tracks the calendar (only low flow measured early, only high flow
    # late), a rating shift is mathematically indistinguishable from the curve's
    # own shape -- no residual test can separate them. Say so instead of
    # reporting a falsely clean "no drift".
    stage_seq = frame[OUT_STAGE].to_numpy(dtype=float)
    st_rank = pd.Series(stage_seq).rank().to_numpy()
    t_rank = pd.Series(t_years).rank().to_numpy()
    with np.errstate(invalid="ignore"):
        stage_time_corr = float(np.corrcoef(st_rank, t_rank)[0, 1])
    result["stage_time_corr"] = stage_time_corr
    confounded = np.isfinite(stage_time_corr) and abs(stage_time_corr) >= 0.75

    tc = t_years - t_years.mean()
    slope = float((tc @ y) / (np.sum(tc ** 2) or 1e-12))
    trend_pct_per_year = (np.exp(slope) - 1.0) * 100.0
    p_trend = _perm_slope_p(t_years, y, slope, rng)

    recent_mask = (dates.iloc[-1] - dates).dt.days.to_numpy() <= _RECENT_WINDOW_DAYS
    min_recent = max(4, n // 4)
    if recent_mask.sum() < min_recent or (~recent_mask).sum() < 3:
        k = min(max(4, n // 3), n - 3)
        recent_mask = np.arange(n) >= (n - k)
    recent = y[recent_mask]
    earlier = y[~recent_mask]
    recent_mean_pct = (np.exp(recent.mean()) - 1.0) * 100.0
    diff_obs = float(recent.mean() - earlier.mean())
    p_recent = _perm_group_p(y, recent_mask, diff_obs, rng)

    # Split-period test: compare an early-half and a late-half fit directly. This
    # sees a shift even when the combined fit absorbed it (seasonal visit
    # schedules tie stage to date and hide it from the residual-vs-time trend).
    stage_all = frame[OUT_STAGE].to_numpy(dtype=float)
    obs_all = frame[OUT_OBS].to_numpy(dtype=float)
    split = _period_split_shift(stage_all, obs_all, np.arange(n), float(fit["h0"]), rng)

    trend_material = abs(trend_pct_per_year) >= TREND_RATE_PCT
    recent_material = abs(recent_mean_pct) >= RECENT_BIAS_PCT
    split_material = split is not None and abs(split["shift_pct"]) >= RECENT_BIAS_PCT

    strong = (
        (trend_material and p_trend < _ALPHA)
        or (recent_material and p_recent < _ALPHA)
        or (split_material and split["p_value"] < _ALPHA)
    )
    weak = (
        (trend_material and p_trend < _ALPHA_WEAK)
        or (recent_material and p_recent < _ALPHA_WEAK)
        or (split_material and split["p_value"] < _ALPHA_WEAK)
    )
    flag = "likely" if strong else "possible" if weak else "none"

    result.update({
        "trend_pct_per_year": float(trend_pct_per_year),
        "trend_p_value": float(p_trend),
        "recent_n": int(recent_mask.sum()),
        "recent_mean_pct": float(recent_mean_pct),
        "recent_p_value": float(p_recent),
        "flag": flag,
    })
    if split is not None:
        result["split_shift_pct"] = split["shift_pct"]
        result["split_p_value"] = split["p_value"]

    if flag == "none" and confounded:
        result["flag"] = "unassessable"
        result["message"] = (
            f"Stage and gauging date are strongly correlated "
            f"(rank r={stage_time_corr:+.2f}) — a rating shift cannot be separated "
            f"from the curve shape for this record. Drift detection is unreliable "
            f"here; schedule gaugings across the flow range each season."
        )
    elif flag == "none":
        result["message"] = "No systematic time trend in the rating-curve residuals."
    else:
        lead = "Likely rating shift" if flag == "likely" else "Possible rating shift"
        bits = []
        if trend_material:
            bits.append(
                f"residuals trend {trend_pct_per_year:+.1f}%/yr (p={p_trend:.3f})"
            )
        if recent_material:
            bits.append(
                f"the last {result['recent_n']} gaugings average {recent_mean_pct:+.0f}% "
                + _direction(recent_mean_pct)
            )
        if split_material:
            bits.append(
                f"a late-period re-fit sits {split['shift_pct']:+.0f}% from an early-period "
                f"re-fit over the shared stage range (p={split['p_value']:.3f})"
            )
        result["message"] = (
            f"{lead}: " + "; ".join(bits) + ". Consider a stage shift or a re-fit on recent gaugings."
        )
    return result
