import numpy as np
import pytest

from rating_curve_automater.piecewise import (
    evaluate_spline,
    fit_piecewise_power_law,
    fit_spline_coef,
)


def _two_control(h0=0.15, bp=0.7, b1=1.3, b2=2.3, n=70, noise=0.04, seed=0):
    rng = np.random.default_rng(seed)
    h = np.sort(rng.uniform(h0 + 0.05, 1.6, n))
    q = np.empty_like(h)
    q[0] = 0.03
    for i in range(1, n):
        b = b1 if h[i] < bp else b2
        q[i] = q[i - 1] * ((h[i] - h0) / (h[i - 1] - h0)) ** b
    q *= np.exp(rng.normal(0, noise, n))
    return h, q


def test_single_segment_matches_a_plain_power_law():
    rng = np.random.default_rng(0)
    h = np.linspace(0.3, 1.4, 40)
    q = 1.1 * (h - 0.15) ** 1.8 * np.exp(rng.normal(0, 0.02, 40))
    model = fit_piecewise_power_law(h, q, 0.15, n_segments=None)

    assert model["is_segmented"] is False
    assert model["a"] == pytest.approx(1.1, rel=0.15)
    assert model["b"] == pytest.approx(1.8, rel=0.1)


def test_forced_two_segments_recovers_exponents_and_is_continuous():
    h, q = _two_control(bp=0.7, b1=1.3, b2=2.3)
    model = fit_piecewise_power_law(h, q, 0.15, n_segments=2)

    assert model["n_segments"] == 2
    lo, hi = model["segments"]
    assert lo["b"] == pytest.approx(1.3, abs=0.25)
    assert hi["b"] == pytest.approx(2.3, abs=0.25)

    bp = model["breakpoints"][0]
    assert abs(evaluate_spline(model["spline"], bp - 1e-4)[0]
               - evaluate_spline(model["spline"], bp + 1e-4)[0]) < 1e-3


def test_auto_stops_at_one_segment_for_a_straight_loglog_line():
    rng = np.random.default_rng(1)
    h = np.linspace(0.3, 1.6, 60)
    q = 0.9 * (h - 0.15) ** 1.9 * np.exp(rng.normal(0, 0.03, 60))
    model = fit_piecewise_power_law(h, q, 0.15, n_segments=None, criterion="bic")
    assert model["is_segmented"] is False


def test_auto_finds_the_kink_when_it_is_real():
    h, q = _two_control(bp=0.7, b1=1.2, b2=2.6, noise=0.05)
    model = fit_piecewise_power_law(h, q, 0.15, n_segments=None, criterion="bic")
    assert model["is_segmented"] is True
    assert model["n_segments"] == 2
    assert 0.55 < model["breakpoints"][0] < 0.9


def test_forcing_more_segments_than_the_data_supports_raises():
    h, q = _two_control(n=12)
    with pytest.raises(ValueError):
        fit_piecewise_power_law(h, q, 0.15, n_segments=4)


def test_weights_shift_the_fit_away_from_flagged_points():
    h, q = _two_control(noise=0.02, n=60)
    q[-6:] *= 0.5  # ruin the top gaugings
    w = np.full(len(h), 1 / 0.03)
    w[-6:] = 1 / 0.6

    unweighted = fit_piecewise_power_law(h, q, 0.15, n_segments=2)
    weighted = fit_piecewise_power_law(h, q, 0.15, weights=w, n_segments=2)

    assert weighted["segments"][-1]["b"] > unweighted["segments"][-1]["b"]


def test_bad_criterion_raises():
    h, q = _two_control()
    with pytest.raises(ValueError):
        fit_piecewise_power_law(h, q, 0.15, criterion="magic")


def test_fit_spline_coef_reduces_to_a_line_with_no_knots():
    u = np.linspace(-2, 1, 30)
    y = 0.4 + 1.7 * u
    coef = fit_spline_coef(u, y, None, [])
    assert coef[0] == pytest.approx(0.4)
    assert coef[1] == pytest.approx(1.7)
