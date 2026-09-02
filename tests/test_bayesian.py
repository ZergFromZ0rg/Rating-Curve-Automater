import numpy as np
import pandas as pd
import pytest

from rating_curve_automater.bayesian import (
    INSTALL_HINT,
    _hs_posterior_summary,
    evaluate_equation,
    is_available,
)
from rating_curve_automater.rating_curve_fitting import fit_rating_curve, predict_discharge

pytestmark = pytest.mark.filterwarnings("ignore")

needs_pymc = pytest.mark.skipif(not is_available(), reason="the 'bayesian' extra (ratingcurve + pymc) is not installed")


def test_method_argument_is_validated():
    df = pd.DataFrame({"stage_m": [0.3, 0.6, 1.0], "discharge_cms": [0.05, 0.3, 1.0], "is_valid": True})
    with pytest.raises(ValueError):
        fit_rating_curve(df, h0=0.18, method="magic")


def test_evaluate_equation_is_a_continuous_piecewise_power_law():
    # single segment: Q = exp(a) * (H - hs0) ** b
    params = {"a": np.log(1.3), "b": [1.8], "hs": [0.15]}
    q = evaluate_equation(params, np.array([0.4, 0.9]))
    assert q == pytest.approx(1.3 * (np.array([0.4, 0.9]) - 0.15) ** 1.8)

    # two segments join continuously (the added term is 0 at its breakpoint)
    two = {"a": np.log(1.0), "b": [1.5, 0.6], "hs": [0.1, 0.7]}
    left = evaluate_equation(two, 0.7 - 1e-6)[0]
    right = evaluate_equation(two, 0.7 + 1e-6)[0]
    assert left == pytest.approx(right, rel=1e-4)


def test_install_hint_names_the_extra():
    assert "rating-curve-automater[bayesian]" in INSTALL_HINT


class _FakeRating:
    """Stands in for a fitted ``ratingcurve`` object — only ``idata.posterior['hs']``."""

    def __init__(self, hs):
        self.idata = type("I", (), {"posterior": {"hs": np.asarray(hs)}})()


def test_hs_posterior_summary_reads_breakpoint_credible_intervals():
    rng = np.random.default_rng(0)
    # (chain, draw, segment, 1): h0 ~ 0.15, one breakpoint ~ 1.0
    h0 = rng.normal(0.15, 0.01, (2, 200, 1, 1))
    bp = rng.normal(1.0, 0.08, (2, 200, 1, 1))
    hs = np.concatenate([h0, bp], axis=2)

    summ = _hs_posterior_summary(_FakeRating(hs), n_seg=2, lo_pct=2.5, hi_pct=97.5)

    lo0, hi0 = summ["h0_ci"]
    assert lo0 < 0.15 < hi0 and (hi0 - lo0) == pytest.approx(0.04, abs=0.02)
    assert len(summ["breakpoint_ci"]) == 1
    lo, hi = summ["breakpoint_ci"][0]
    assert lo < 1.0 < hi and (hi - lo) == pytest.approx(0.31, abs=0.1)
    assert summ["breakpoint_sd"][0] == pytest.approx(0.08, abs=0.03)


def test_hs_posterior_summary_single_segment_has_no_breakpoints():
    hs = np.full((2, 50, 1, 1), 0.2)
    summ = _hs_posterior_summary(_FakeRating(hs), n_seg=1, lo_pct=2.5, hi_pct=97.5)
    assert summ["breakpoint_ci"] == [] and summ["breakpoint_sd"] == []
    assert summ["h0_ci"] == (pytest.approx(0.2), pytest.approx(0.2))


@needs_pymc
def test_bayesian_fit_recovers_a_single_power_law_and_agrees_with_ols():
    rng = np.random.default_rng(0)
    h = np.sort(rng.uniform(0.3, 1.5, 45))
    q = 1.2 * (h - 0.15) ** 1.75 * np.exp(rng.normal(0, 0.05, 45))
    df = pd.DataFrame({"stage_m": h, "discharge_cms": q, "is_valid": True})

    bayes = fit_rating_curve(df, method="bayesian", bayesian_sampler="advi", random_state=0)
    ols = fit_rating_curve(df, method="ols", n_bootstrap=300, random_state=0)

    assert bayes["method"] == "bayesian"
    assert bayes["b"] == pytest.approx(1.75, abs=0.2)
    assert bayes["h0"] == pytest.approx(0.15, abs=0.1)
    assert bayes["r_squared"] > 0.95
    # the two estimators should land close together
    assert bayes["b"] == pytest.approx(ols["b"], abs=0.15)
    assert predict_discharge(bayes, np.array([0.5, 1.0])).shape == (2,)


@needs_pymc
def test_bayesian_sampler_selection():
    from rating_curve_automater.bayesian import _resolve_sampler

    assert _resolve_sampler("auto", 50) == "nuts"
    assert _resolve_sampler("auto", 5000) == "advi"
    assert _resolve_sampler("advi", 50) == "advi"
    with pytest.raises(ValueError):
        _resolve_sampler("gibbs", 50)

    rng = np.random.default_rng(4)
    h = np.sort(rng.uniform(0.3, 1.5, 40))
    q = 1.2 * (h - 0.15) ** 1.8 * np.exp(rng.normal(0, 0.05, 40))
    df = pd.DataFrame({"stage_m": h, "discharge_cms": q, "is_valid": True})
    fit = fit_rating_curve(df, method="bayesian", bayesian_sampler="nuts", random_state=0)
    assert fit["bayes"]["sampler"] == "nuts"


@needs_pymc
def test_bayesian_fit_bands_come_from_the_posterior():
    rng = np.random.default_rng(1)
    h = np.sort(rng.uniform(0.3, 1.5, 40))
    q = 1.1 * (h - 0.12) ** 1.9 * np.exp(rng.normal(0, 0.06, 40))
    df = pd.DataFrame({"stage_m": h, "discharge_cms": q, "is_valid": True})

    fit = fit_rating_curve(df, method="bayesian", bayesian_sampler="advi", random_state=0)
    bands = fit["bands"]

    assert bands is not None
    assert bands["kind"] == "posterior"
    assert bands["b_ci"] is not None
    qg = np.asarray(bands["q"])
    assert np.all(bands["ci_lower"] <= qg + 1e-9) and np.all(qg <= bands["ci_upper"] + 1e-9)
    assert np.all(np.asarray(bands["pi_lower"]) <= np.asarray(bands["ci_lower"]) + 1e-9)


@needs_pymc
def test_bayesian_auto_segments_picks_one_for_a_clean_power_law():
    rng = np.random.default_rng(2)
    h = np.sort(rng.uniform(0.3, 1.5, 35))
    q = 1.0 * (h - 0.15) ** 1.8 * np.exp(rng.normal(0, 0.05, 35))
    df = pd.DataFrame({"stage_m": h, "discharge_cms": q, "is_valid": True})

    fit = fit_rating_curve(df, method="bayesian", segments="auto", bayesian_sampler="advi", random_state=0)
    assert fit["n_segments"] == 1
    assert fit["segment_selection"] == "auto"
    assert set(fit["bayes"]["segment_bic"]) == {1, 2, 3}
    assert "BIC picked 1" in fit["bayes"]["auto_segments_note"]


@needs_pymc
def test_bayesian_auto_segments_picks_two_for_a_compound_channel():
    rng = np.random.default_rng(5)
    h = np.sort(rng.uniform(0.25, 2.0, 60))
    x = np.maximum(h - 0.15, 1e-9)
    a_high = (6.0 * (0.85) ** 1.5) / (0.85) ** 2.6
    q = np.where(h < 1.0, 6.0 * x ** 1.5, a_high * x ** 2.6)
    q *= np.exp(rng.normal(0, 0.04, h.size))
    df = pd.DataFrame({"stage_m": h, "discharge_cms": q, "is_valid": True})

    fit = fit_rating_curve(df, method="bayesian", segments="auto", bayesian_sampler="advi", random_state=0)
    assert fit["n_segments"] >= 2
    assert fit["is_segmented"]


@needs_pymc
def test_bayesian_reports_h0_credible_interval():
    rng = np.random.default_rng(0)
    h = np.sort(rng.uniform(0.3, 1.5, 40))
    q = 1.2 * (h - 0.15) ** 1.8 * np.exp(rng.normal(0, 0.05, 40))
    df = pd.DataFrame({"stage_m": h, "discharge_cms": q, "is_valid": True})

    fit = fit_rating_curve(df, method="bayesian", bayesian_sampler="advi", random_state=0)
    lo, hi = fit["h0_ci"]
    assert lo <= fit["h0"] <= hi
    assert fit["h0_sd"] > 0
    assert fit["bands"]["h0_ci"] == pytest.approx(fit["h0_ci"])
    assert fit["bands"]["a_ci"] is not None
    assert "breakpoint_ci" not in fit          # single segment


@needs_pymc
def test_bayesian_segmented_fit_brackets_each_breakpoint():
    rng = np.random.default_rng(5)
    h = np.sort(rng.uniform(0.25, 2.0, 70))
    x = np.maximum(h - 0.15, 1e-9)
    a_high = (6.0 * (0.9) ** 1.4) / (0.9) ** 2.7
    q = np.where(h < 1.0, 6.0 * x ** 1.4, a_high * x ** 2.7)
    q *= np.exp(rng.normal(0, 0.035, h.size))
    df = pd.DataFrame({"stage_m": h, "discharge_cms": q, "is_valid": True})

    fit = fit_rating_curve(df, method="bayesian", segments=2, bayesian_sampler="nuts", random_state=0)

    assert len(fit["breakpoint_ci"]) == len(fit["breakpoints"])
    for bp, (lo, hi) in zip(fit["breakpoints"], fit["breakpoint_ci"]):
        assert lo <= bp <= hi
        assert hi > lo
    assert fit["bands"]["breakpoint_ci"] == fit["breakpoint_ci"]
    assert fit["segments"][1]["breakpoint_ci"] == fit["breakpoint_ci"][0]
