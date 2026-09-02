import numpy as np
import pandas as pd

from rating_curve_automater.rating_curve_fitting import fit_rating_curve
from rating_curve_automater.rating_curve_uncertainty import bootstrap_rating_curve


def _noisy_curve(h0=0.18, a=1.2, b=1.7, n=50, noise=0.08, seed=1):
    rng = np.random.default_rng(seed)
    h = np.linspace(h0 + 0.05, 1.3, n)
    q = a * (h - h0) ** b
    q *= np.exp(rng.normal(0.0, noise, size=n))
    return pd.DataFrame({"stage_m": h, "discharge_cms": q, "is_valid": True})


def _bands(df, **kw):
    kw.setdefault("h0", 0.18)
    kw.setdefault("n_bootstrap", 400)
    kw.setdefault("random_state", 0)
    fit = fit_rating_curve(df, **kw)
    return fit, fit["bands"]


def test_bands_are_nested_and_ordered():
    _, bands = _bands(_noisy_curve())

    assert bands is not None
    q = np.asarray(bands["q"])
    assert np.all(bands["ci_lower"] <= q + 1e-9)
    assert np.all(q <= bands["ci_upper"] + 1e-9)
    assert np.all(bands["pi_lower"] <= bands["ci_lower"] + 1e-9)
    assert np.all(bands["pi_upper"] >= bands["ci_upper"] - 1e-9)


def test_prediction_band_is_wider_than_confidence_band():
    _, bands = _bands(_noisy_curve(noise=0.12))

    ci_width = np.asarray(bands["ci_upper"]) - np.asarray(bands["ci_lower"])
    pi_width = np.asarray(bands["pi_upper"]) - np.asarray(bands["pi_lower"])
    assert np.mean(pi_width) > np.mean(ci_width) * 1.5


def test_more_scatter_gives_a_wider_confidence_band():
    _, tight = _bands(_noisy_curve(noise=0.03, seed=2))
    _, loose = _bands(_noisy_curve(noise=0.15, seed=2))

    def rel_width(b):
        return np.mean((np.asarray(b["ci_upper"]) - np.asarray(b["ci_lower"])) / np.asarray(b["q"]))

    assert rel_width(loose) > rel_width(tight)


def test_b_confidence_interval_brackets_the_true_exponent():
    _, bands = _bands(_noisy_curve(b=1.7, noise=0.05, n=60), n_bootstrap=600)

    lo, hi = bands["b_ci"]
    assert lo < 1.7 < hi
    assert hi - lo < 0.6  # not uselessly wide


def test_bootstrap_is_deterministic_with_a_seed():
    df = _noisy_curve()
    a = fit_rating_curve(df, h0=0.18, n_bootstrap=300, random_state=42)["bands"]
    b = fit_rating_curve(df, h0=0.18, n_bootstrap=300, random_state=42)["bands"]
    assert np.array_equal(a["ci_lower"], b["ci_lower"])
    assert a["b_ci"] == b["b_ci"]


def test_no_bands_below_four_points():
    df = pd.DataFrame({
        "stage_m": [0.3, 0.6, 1.0],
        "discharge_cms": [0.05, 0.3, 1.0],
        "is_valid": True,
    })
    fit = fit_rating_curve(df, h0=0.18, n_bootstrap=500)
    assert fit["bands"] is None


def test_default_fit_has_no_bands_unless_requested():
    fit = fit_rating_curve(_noisy_curve(), h0=0.18)
    assert "bands" not in fit or fit["bands"] is None


def test_segmented_fit_gets_curve_bands_but_no_scalar_param_ci():
    low = np.linspace(0.25, 0.6, 30)
    high = np.linspace(0.6, 1.3, 30)
    q_low = 0.5 * (low - 0.18) ** 1.3
    q_break = 0.5 * (0.6 - 0.18) ** 1.3
    a_high = q_break / (0.6 - 0.18) ** 2.4
    q_high = a_high * (high - 0.18) ** 2.4
    df = pd.DataFrame({
        "stage_m": np.concatenate([low, high]),
        "discharge_cms": np.concatenate([q_low, q_high]),
        "is_valid": True,
    })

    fit = fit_rating_curve(df, h0=0.18, segments=2, n_bootstrap=300, random_state=0)
    bands = fit["bands"]
    assert bands is not None
    assert bands["a_ci"] is None and bands["b_ci"] is None
    assert np.all(np.asarray(bands["pi_lower"]) <= np.asarray(bands["ci_lower"]) + 1e-9)
    q = np.asarray(bands["q"])
    assert np.all(bands["ci_lower"] <= q + 1e-9) and np.all(q <= bands["ci_upper"] + 1e-9)


def test_estimated_h0_is_propagated_through_the_band():
    df = _noisy_curve(h0=0.18, noise=0.08, n=55)

    fixed = fit_rating_curve(df, h0=0.18, n_bootstrap=500, random_state=0)["bands"]
    est = fit_rating_curve(df, n_bootstrap=500, random_state=0)["bands"]

    # Fixed h0 -> nothing to propagate; estimated h0 -> a reported interval.
    assert fixed["h0_ci"] is None and fixed["h0_reestimated"] is False
    assert est["h0_reestimated"] is True
    lo, hi = est["h0_ci"]
    assert lo < hi and hi - lo > 0.0

    # The extra parameter uncertainty widens the confidence band at the low end,
    # where h0 matters most.
    low = np.asarray(est["stage"]) < np.median(est["stage"])
    def rel_low(b):
        w = (np.asarray(b["ci_upper"]) - np.asarray(b["ci_lower"])) / np.asarray(b["q"])
        return float(np.mean(w[low]))
    assert rel_low(est) > rel_low(fixed)


def test_direct_helper_matches_fit_wiring():
    df = _noisy_curve()
    fit = fit_rating_curve(df, h0=0.18)
    stage = df["stage_m"].to_numpy()
    disch = df["discharge_cms"].to_numpy()
    weights = np.full(len(df), 1.0 / 0.07)

    bands = bootstrap_rating_curve(stage, disch, fit, weights, n_bootstrap=200, random_state=7)
    assert set(bands) >= {"ci_lower", "ci_upper", "pi_lower", "pi_upper", "b_ci", "level"}
