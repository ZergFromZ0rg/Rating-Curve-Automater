import numpy as np
import pandas as pd
import pytest

from rating_curve_automater.rating_curve_fitting import fit_rating_curve, predict_discharge
from rating_curve_automater.rating_table import (
    OUT_IN_RANGE,
    OUT_Q,
    OUT_STAGE,
    build_rating_table,
    export_rating_table_csv,
)


def _fit(noise=0.0, n=40, **kw):
    rng = np.random.default_rng(0)
    h = np.linspace(0.25, 1.3, n)
    q = 1.2 * (h - 0.18) ** 1.7 * np.exp(rng.normal(0, noise, n))
    df = pd.DataFrame({"stage_m": h, "discharge_cms": q, "is_valid": True})
    return fit_rating_curve(df, h0=0.18, **kw)


def test_table_grid_spacing_and_values():
    fit = _fit()
    t = build_rating_table(fit, step=0.05)

    stages = t[OUT_STAGE].to_numpy()
    assert np.allclose(np.diff(stages), 0.05)
    assert (stages > fit["h0"]).all()
    assert np.allclose(t[OUT_Q].to_numpy(), predict_discharge(fit, stages))
    assert (t[OUT_Q].diff().dropna() > 0).all()  # monotincreasing


def test_default_range_is_the_gauged_range():
    fit = _fit()
    t = build_rating_table(fit, step=0.01)
    assert t[OUT_STAGE].min() <= fit["stage_min"] < t[OUT_STAGE].min() + 0.01
    assert fit["stage_max"] - 0.01 < t[OUT_STAGE].max() <= fit["stage_max"] + 0.01
    assert t[OUT_IN_RANGE].all()


def test_extrapolated_rows_are_flagged():
    fit = _fit()
    t = build_rating_table(fit, step=0.05, stage_min=0.20, stage_max=2.0)

    assert not t[OUT_IN_RANGE].all()
    below = t[t[OUT_STAGE] < fit["stage_min"]]
    above = t[t[OUT_STAGE] > fit["stage_max"]]
    assert (~below[OUT_IN_RANGE]).all()
    assert (~above[OUT_IN_RANGE]).all()


def test_band_columns_present_and_blank_outside_bootstrap_grid():
    fit = _fit(noise=0.08, n=50, n_bootstrap=300, random_state=0)
    t = build_rating_table(fit, step=0.05, stage_max=2.0)

    ci_lo = [c for c in t.columns if "confidence lower" in c][0]
    ci_hi = [c for c in t.columns if "confidence upper" in c][0]
    within = t[t[OUT_STAGE] <= fit["stage_max"]]
    beyond = t[t[OUT_STAGE] > fit["bands"]["stage"].max()]

    assert within[ci_lo].notna().all()
    assert (within[ci_lo] <= within[OUT_Q] + 1e-9).all()
    assert (within[ci_hi] >= within[OUT_Q] - 1e-9).all()
    assert beyond[ci_lo].isna().all()


def test_no_band_columns_when_fit_has_no_bands():
    t = build_rating_table(_fit(), step=0.1)
    assert not any("confidence" in c for c in t.columns)


def test_step_must_be_positive():
    with pytest.raises(ValueError):
        build_rating_table(_fit(), step=0)


def test_absurd_row_count_is_refused():
    with pytest.raises(ValueError):
        build_rating_table(_fit(), step=1e-7)


def test_segmented_fit_table_has_the_kink():
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
    fit = fit_rating_curve(df, h0=0.18, segments=2)

    t = build_rating_table(fit, step=0.02)
    assert np.allclose(t[OUT_Q].to_numpy(), predict_discharge(fit, t[OUT_STAGE].to_numpy()))


def test_csv_round_trips(tmp_path):
    fit = _fit()
    out = export_rating_table_csv(fit, tmp_path / "rt.csv", step=0.05)
    assert out.exists()
    back = pd.read_csv(out)
    assert list(back.columns)[:2] == [OUT_STAGE, OUT_Q]
    assert len(back) == len(build_rating_table(fit, step=0.05))
