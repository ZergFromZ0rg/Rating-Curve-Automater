import numpy as np
import pandas as pd
import pytest

from rating_curve_automater.bayesian import (
    INSTALL_HINT,
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
