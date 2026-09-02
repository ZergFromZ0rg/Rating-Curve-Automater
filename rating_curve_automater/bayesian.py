"""Bayesian rating-curve backend — a thin wrapper over thodson-usgs
``ratingcurve`` (a PyMC multi-segment power-law model).

Optional; enable with::

    pip install "rating-curve-automater[bayesian]"

:func:`fit_bayesian_rating_curve` returns the same fit dict shape the
least-squares path produces (``a`` / ``b`` / ``breakpoints`` / ``segments`` /
``equation`` / ``bands`` …) so the plot, report, rating table and drift check all
work unchanged. The fitted curve is evaluated deterministically from the
posterior-mean power-law equation; confidence and prediction bands come from the
posterior instead of the wild bootstrap.

Because ``ratingcurve`` samples the breakpoint stages ``hs`` jointly with the
power-law coefficients, their uncertainty is already in the posterior. The fit
dict exposes it: ``h0_ci`` / ``h0_sd`` always, and ``breakpoint_ci`` /
``breakpoint_sd`` (plus a per-segment ``breakpoint_ci``) for a multi-segment
fit. ``assess_fit`` warns when a breakpoint's credible interval is wide relative
to the gauged stage range, the plot shades it, and the report lists it.
"""

from __future__ import annotations

import warnings

import numpy as np

from rating_curve_automater.piecewise import segment_r_squared

INSTALL_HINT = (
    "The Bayesian backend needs the optional 'bayesian' extra:\n"
    '    pip install "rating-curve-automater[bayesian]"\n'
    "(installs ratingcurve + PyMC)."
)


def is_available() -> bool:
    try:
        import ratingcurve.ratings  # noqa: F401
    except Exception:
        return False
    return True


def _power_law_rating():
    try:
        from ratingcurve.ratings import PowerLawRating
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(INSTALL_HINT) from exc
    return PowerLawRating


def evaluate_equation(params: dict, stage) -> np.ndarray:
    """Discharge from a fitted ``{a, b[], hs[]}`` equation (``ratingcurve``'s
    denormalised power law), for any stage. ``hs[0]`` is the stage of zero flow."""
    a = float(params["a"])
    b = np.asarray(params["b"], dtype=float)
    hs = np.asarray(params["hs"], dtype=float)
    stage = np.atleast_1d(np.asarray(stage, dtype=float))
    ho = np.ones(b.size)
    ho[0] = 0.0
    log_q = np.full(stage.shape, a, dtype=float)
    with np.errstate(divide="ignore"):
        for i in range(b.size):
            arg = np.clip(stage - hs[i], 0.0, None) + ho[i]
            log_q = log_q + b[i] * np.log(np.maximum(arg, 1e-12))
    return np.exp(log_q)


def _segment_slopes(b: np.ndarray) -> np.ndarray:
    """``ratingcurve`` stores b[0] as the base slope and b[i>0] as deviations."""
    return np.cumsum(np.asarray(b, dtype=float))


def _hs_posterior_summary(rating, n_seg: int, lo_pct: float, hi_pct: float) -> dict:
    """Credible intervals for the jointly-sampled breakpoint stages ``hs``.

    ``hs[0]`` is the stage of zero flow (h0); ``hs[1:]`` are the segment
    breakpoints. ``ratingcurve`` samples them inside the power-law model, so
    their uncertainty is already in the posterior — this just reads it out.
    Returns ``{"h0_ci", "h0_sd", "breakpoint_ci", "breakpoint_sd"}`` with the
    breakpoint entries lists (empty for a single-segment fit).
    """
    hs = np.asarray(rating.idata.posterior["hs"]).reshape(-1, n_seg)  # (draws, seg)

    def _ci(col: np.ndarray) -> tuple[float, float]:
        lo, hi = np.percentile(col, [lo_pct, hi_pct])
        return float(lo), float(hi)

    return {
        "h0_ci": _ci(hs[:, 0]),
        "h0_sd": float(np.std(hs[:, 0])),
        "breakpoint_ci": [_ci(hs[:, i]) for i in range(1, n_seg)],
        "breakpoint_sd": [float(np.std(hs[:, i])) for i in range(1, n_seg)],
    }


def _posterior_mu_draws(rating, grid: np.ndarray) -> np.ndarray:
    """(draws, grid) posterior samples of the *mean* curve (parameter
    uncertainty only), reconstructed from the denormalised equation."""
    post = rating.idata.posterior
    a_z = np.asarray(post["a"]).reshape(-1)
    b_z = np.asarray(post["b"]).reshape(a_z.size, -1)
    hs = np.asarray(post["hs"]).reshape(a_z.size, -1)
    mean_ = float(rating.q_transform.mean_)
    std_ = float(rating.q_transform.std_)

    a = mean_ + std_ * a_z                       # (draws,)
    b = std_ * b_z                               # (draws, seg)
    ho = np.ones(b.shape[1])
    ho[0] = 0.0
    # (draws, seg, grid)
    arg = np.clip(grid[None, None, :] - hs[:, :, None], 0.0, None) + ho[None, :, None]
    with np.errstate(divide="ignore"):
        u = np.log(np.maximum(arg, 1e-12))
    log_mu = a[:, None] + np.einsum("ds,dsg->dg", b, u)
    return np.exp(log_mu)


#: At or below this many gaugings, NUTS (exact posterior sampling) is fast enough
#: and clearly more accurate than ADVI's variational approximation -- the ADVI
#: fit misplaced breakpoints in benchmarking. Above it, ``sampler="auto"`` uses
#: ADVI to keep the fit interactive.
NUTS_MAX_POINTS = 200


def _resolve_sampler(sampler: str, n_points: int) -> str:
    if sampler not in ("auto", "nuts", "advi"):
        raise ValueError("sampler must be 'auto', 'nuts' or 'advi'.")
    if sampler != "auto":
        return sampler
    return "nuts" if n_points <= NUTS_MAX_POINTS else "advi"


#: Most segments ``segments="auto"`` will try for the Bayesian backend (each
#: extra segment is another full NUTS fit).
BAYES_MAX_SEGMENTS = 3


def _bayes_bic(params: dict, stage: np.ndarray, discharge: np.ndarray, n_seg: int) -> float:
    """BIC of a fitted Bayesian curve, judged on its posterior-mean equation in
    log space -- the same yardstick the least-squares piecewise path uses, so
    ``segments="auto"`` picks a segment count it can justify."""
    modelled = evaluate_equation(params, stage)
    h0 = float(np.asarray(params["hs"], dtype=float)[0])
    keep = (stage - h0 > 0) & (discharge > 0) & (modelled > 0)
    n = int(keep.sum())
    if n < 2 * n_seg + 2:
        return float("inf")
    rss = float(np.sum((np.log(discharge[keep]) - np.log(modelled[keep])) ** 2))
    rss = max(rss, 1e-300)
    k = 2 * n_seg + 1  # a, per-segment slope + breakpoint, sigma
    return n * np.log(rss / n) + k * np.log(n)


def fit_bayesian_rating_curve(
    stage: np.ndarray,
    discharge: np.ndarray,
    uncertainty_frac: np.ndarray,
    *,
    segments: int | str = 1,
    level: float = 0.95,
    random_state: int | None = None,
    draws: int | None = None,
    sampler: str = "auto",
    max_segments: int = BAYES_MAX_SEGMENTS,
) -> dict:
    """Fit a rating curve with ``ratingcurve``'s Bayesian power-law model.

    ``sampler`` is ``"nuts"`` (exact, slower), ``"advi"`` (variational, fast) or
    ``"auto"`` -- NUTS for up to :data:`NUTS_MAX_POINTS` gaugings, ADVI above.
    ``segments="auto"`` fits 1..``max_segments`` segments and keeps the count
    with the lowest BIC (each candidate is a full fit, so this is slow).
    """
    PowerLawRating = _power_law_rating()

    stage = np.asarray(stage, dtype=float)
    discharge = np.asarray(discharge, dtype=float)
    q_sigma = np.asarray(uncertainty_frac, dtype=float) * discharge

    auto = isinstance(segments, str)
    n_seg = 1 if auto else max(1, int(segments))

    sampler = _resolve_sampler(sampler, int(stage.size))
    if draws is None:
        draws = 1500 if sampler == "nuts" else 4000
    fit_kwargs = dict(method=sampler, progressbar=False,
                      random_seed=random_state, draws=draws)
    if sampler == "nuts":
        fit_kwargs.update(tune=1000, chains=4)

    def _fit(n: int):
        r = PowerLawRating(segments=n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r.fit(h=stage, q=discharge, q_sigma=q_sigma, **fit_kwargs)
        return r, r.equation()

    if auto:
        cap = max(1, min(int(max_segments), (stage.size - 2) // 4))
        candidates = []
        for n in range(1, cap + 1):
            try:
                r, p = _fit(n)
            except Exception:
                continue
            candidates.append((_bayes_bic(p, stage, discharge, n), n, r, p))
        if not candidates:
            raise RuntimeError("Bayesian auto-segment fit did not converge for any segment count.")
        bic_by_n = {n: bic for bic, n, _, _ in candidates}
        _, n_seg, rating, params = min(candidates, key=lambda t: t[0])
        auto_bic = bic_by_n
    else:
        rating, params = _fit(n_seg)
        auto_bic = None
    b_terms = np.asarray(params["b"], dtype=float)
    hs = np.asarray(params["hs"], dtype=float)
    slopes = _segment_slopes(b_terms)
    h0 = float(hs[0])
    breakpoints = [float(x) for x in hs[1:]]

    modelled = evaluate_equation(params, stage)
    keep = (stage - h0 > 0) & (discharge > 0)

    def _r2(weights: np.ndarray | None) -> float:
        obs, pred = discharge[keep], modelled[keep]
        pw = np.ones_like(obs) if weights is None else weights[keep] ** 2
        q_bar = float(np.sum(pw * obs) / np.sum(pw))
        ss_res = float(np.sum(pw * (obs - pred) ** 2))
        ss_tot = float(np.sum(pw * (obs - q_bar) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot else 1.0

    weights = 1.0 / np.clip(np.asarray(uncertainty_frac, dtype=float), 1e-6, None)

    grid = np.linspace(stage[keep].min(), stage[keep].max(), 100)
    q_grid = evaluate_equation(params, grid)
    lo_pct, hi_pct = 100.0 * (1.0 - level) / 2.0, 100.0 * (1.0 + level) / 2.0

    mu_draws = _posterior_mu_draws(rating, grid)
    ci_lower, ci_upper = np.percentile(mu_draws, [lo_pct, hi_pct], axis=0)
    pp = np.asarray(rating.predict_posterior(grid))          # (grid, draws)
    pi_lower, pi_upper = np.percentile(pp, [lo_pct, hi_pct], axis=1)

    ci_lower = np.minimum(ci_lower, q_grid)
    ci_upper = np.maximum(ci_upper, q_grid)
    pi_lower = np.minimum(pi_lower, ci_lower)
    pi_upper = np.maximum(pi_upper, ci_upper)

    n_draws = mu_draws.shape[0]
    median_stage = float(np.median(stage[keep]))
    q_at_median = float(np.interp(median_stage, grid, q_grid))
    ci_hw = float(np.interp(median_stage, grid, (ci_upper - ci_lower) / 2.0))

    if n_seg == 1:
        b_draws = float(rating.q_transform.std_) * np.asarray(rating.idata.posterior["b"]).reshape(-1)
        b_ci = tuple(float(v) for v in np.percentile(b_draws, [lo_pct, hi_pct]))
        a_z = np.asarray(rating.idata.posterior["a"]).reshape(-1)
        a_draws = np.exp(float(rating.q_transform.mean_) + float(rating.q_transform.std_) * a_z)
        a_ci = tuple(float(v) for v in np.percentile(a_draws, [lo_pct, hi_pct]))
    else:
        b_ci = None
        a_ci = None

    hs_post = _hs_posterior_summary(rating, n_seg, lo_pct, hi_pct)

    bands = {
        "kind": "posterior",
        "level": float(level),
        "n_bootstrap": n_draws,
        "n_success": n_draws,
        "stage": grid,
        "q": q_grid,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "pi_lower": pi_lower,
        "pi_upper": pi_upper,
        "a_ci": a_ci,
        "b_ci": b_ci,
        "h0_ci": hs_post["h0_ci"],
        "breakpoint_ci": hs_post["breakpoint_ci"] or None,
        "ci_halfwidth_pct_at_median": 100.0 * ci_hw / q_at_median if q_at_median > 0 else float("nan"),
    }

    def _seg_record(i: int, lo: float, hi: float) -> dict:
        in_seg = (stage >= lo) & (stage < hi) & keep
        seg_r2 = segment_r_squared(discharge[in_seg], modelled[in_seg])
        # local a for Q = a*(H-h0)^b_seg on this segment
        a_local = float(np.exp(np.log(max(evaluate_equation(params, (lo + hi) / 2)[0], 1e-12))
                               - slopes[i] * np.log(max((lo + hi) / 2 - h0, 1e-9))))
        return {
            "a": a_local,
            "b": float(slopes[i]),
            "r_squared": seg_r2,
            "n_points": int(in_seg.sum()),
            "stage_min": float(lo if np.isfinite(lo) else stage[keep].min()),
            "stage_max": float(hi if np.isfinite(hi) else stage[keep].max()),
        }

    edges = [stage[keep].min(), *breakpoints, stage[keep].max()]
    segment_records = [_seg_record(i, edges[i], edges[i + 1]) for i in range(n_seg)]

    if n_seg == 1:
        equation = f"Q = {segment_records[0]['a']:.6f} * (H - {h0:.3f})^{slopes[0]:.6f}"
    else:
        parts = []
        for i, seg in enumerate(segment_records):
            lo = breakpoints[i - 1] if i > 0 else None
            hi = breakpoints[i] if i < len(breakpoints) else None
            rng = (f"H < {hi:.3f}" if hi is not None and lo is None
                   else f"H >= {lo:.3f}" if hi is None
                   else f"{lo:.3f} <= H < {hi:.3f}")
            parts.append(f"{rng}: Q = {seg['a']:.6f} * (H - {h0:.3f})^{seg['b']:.6f}")
        equation = "; ".join(parts)

    fit: dict = {
        "method": "bayesian",
        "bayes_params": {"a": float(params["a"]), "b": b_terms.tolist(), "hs": hs.tolist()},
        "is_segmented": n_seg > 1,
        "h0": h0,
        "h0_estimated": True,
        "n_points": int(keep.sum()),
        "n_segments": n_seg,
        "segments": segment_records if n_seg > 1 else None,
        "a": segment_records[0]["a"],
        "b": float(slopes[0]),
        "r_squared": _r2(None),
        "r_squared_weighted": _r2(weights),
        "equation": equation,
        "bands": bands,
        "h0_ci": hs_post["h0_ci"],
        "h0_sd": hs_post["h0_sd"],
        "bayes": {
            "sampler": sampler,
            "draws": n_draws,
            "segment_selection": "auto" if auto else "forced",
            "segment_bic": ({int(k): round(float(v), 2) for k, v in auto_bic.items()}
                            if auto_bic else None),
            "auto_segments_note": (
                f"segments='auto': fitted 1..{max(auto_bic)} segments, BIC picked "
                f"{n_seg}." if auto_bic else None
            ),
        },
    }
    if auto:
        fit["segment_selection"] = "auto"
        fit["criterion"] = "bic"
    if n_seg > 1:
        fit["breakpoints"] = breakpoints
        fit["breakpoint"] = breakpoints[0]
        fit["breakpoint_ci"] = hs_post["breakpoint_ci"]
        fit["breakpoint_sd"] = hs_post["breakpoint_sd"]
        for seg, (lo, hi) in zip(fit["segments"][1:], hs_post["breakpoint_ci"]):
            seg["breakpoint_ci"] = (lo, hi)
    else:
        fit.pop("segments")
    return fit
