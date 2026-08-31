import numpy as np
import pandas as pd
import pytest

from rating_curve_automater.rating_curve_drift import (
    OUT_PCT,
    OUT_SHIFT,
    assess_temporal_drift,
    build_residual_frame,
)
from rating_curve_automater.rating_curve_fitting import fit_rating_curve


def _dated_frame(drift_per_year=0.0, years=8, n=60, noise=0.04, seed=1, h0=0.15):
    rng = np.random.default_rng(seed)
    days = np.sort(rng.uniform(0, years * 365, n))
    dates = pd.Timestamp("2015-01-01") + pd.to_timedelta(days, unit="D")
    t_years = days / 365.25
    h = rng.uniform(h0 + 0.1, 1.4, n)
    q = 1.2 * (h - h0) ** 1.75 * np.exp(np.log1p(drift_per_year) * t_years) * np.exp(rng.normal(0, noise, n))
    return pd.DataFrame({"date": dates, "stage_m": h, "discharge_cms": q, "is_valid": True})


def test_no_drift_data_is_flagged_none():
    fit = fit_rating_curve(_dated_frame(drift_per_year=0.0, seed=2), h0=0.15, random_state=0)
    assert fit["drift"]["flag"] == "none"
    assert "No systematic time trend" in fit["drift"]["message"]


def test_scour_drift_is_detected_and_quantified():
    fit = fit_rating_curve(_dated_frame(drift_per_year=0.045, seed=3), h0=0.15, random_state=0)
    drift = fit["drift"]

    assert drift["flag"] == "likely"
    assert drift["trend_pct_per_year"] == pytest.approx(4.5, abs=1.5)
    assert drift["trend_p_value"] < 0.05
    assert drift["recent_mean_pct"] > 0            # recent gaugings sit above the curve
    assert "scour" in drift["message"] or "above the curve" in drift["message"]


def test_aggradation_drift_direction():
    fit = fit_rating_curve(_dated_frame(drift_per_year=-0.05, seed=4), h0=0.15, random_state=0)
    drift = fit["drift"]
    assert drift["flag"] in ("likely", "possible")
    assert drift["trend_pct_per_year"] < 0
    assert drift["recent_mean_pct"] < 0


def test_undated_gaugings_give_no_drift_block():
    df = pd.DataFrame({
        "stage_m": np.linspace(0.3, 1.4, 30),
        "discharge_cms": 1.1 * (np.linspace(0.3, 1.4, 30) - 0.15) ** 1.7,
        "is_valid": True,
    })
    fit = fit_rating_curve(df, h0=0.15, random_state=0)
    assert "drift" not in fit


def test_too_short_a_record_is_reported_but_not_flagged():
    df = _dated_frame(drift_per_year=0.2, n=10)
    df["date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(np.arange(10), unit="D")
    fit = fit_rating_curve(df, h0=0.15, random_state=0)
    drift = fit["drift"]
    assert drift["flag"] == "none"
    assert "too short" in drift["message"]


def test_residual_frame_has_percent_and_stage_shift():
    df = _dated_frame(drift_per_year=0.03)
    fit = fit_rating_curve(df, h0=0.15, random_state=0)
    frame = build_residual_frame(df.pipe(lambda d: d), fit)  # canonical already
    assert frame is not None
    assert OUT_PCT in frame.columns and OUT_SHIFT in frame.columns
    assert frame["Date"].is_monotonic_increasing
    # a positive discharge residual corresponds to a positive stage shift
    same_sign = np.sign(frame[OUT_PCT]) == np.sign(frame[OUT_SHIFT])
    assert same_sign.mean() > 0.9


def test_drift_is_reproducible_with_a_seed():
    df = _dated_frame(drift_per_year=0.04)
    a = fit_rating_curve(df, h0=0.15, random_state=7)["drift"]
    b = fit_rating_curve(df, h0=0.15, random_state=7)["drift"]
    assert a["trend_p_value"] == b["trend_p_value"]
    assert a["recent_p_value"] == b["recent_p_value"]


def test_assess_returns_none_without_dates():
    df = pd.DataFrame({"stage_m": [0.3, 0.6, 1.0, 1.2], "discharge_cms": [0.1, 0.4, 1.0, 1.4]})
    assert assess_temporal_drift(df, {"h0": 0.15, "a": 1.0, "b": 1.7, "is_segmented": False}) is None


def test_split_period_catches_a_step_shift_the_combined_fit_absorbs():
    # Dates decoupled from stage; a +15% step change halfway through the record.
    rng = np.random.default_rng(11)
    n = 70
    days = np.sort(rng.uniform(0, 6 * 365, n))
    dates = pd.Timestamp("2016-01-01") + pd.to_timedelta(days, unit="D")
    h = rng.uniform(0.30, 1.8, n)
    q = 6.0 * (h - 0.15) ** 1.9 * np.exp(rng.normal(0, 0.04, n))
    q[days >= np.median(days)] *= 1.15
    df = pd.DataFrame({"date": dates, "stage_m": h, "discharge_cms": q, "is_valid": True})

    drift = fit_rating_curve(df, segments="auto", random_state=0)["drift"]
    assert drift["flag"] == "likely"
    assert abs(drift["split_shift_pct"]) > 8
    assert drift["split_p_value"] < 0.05


def test_stage_confounded_with_date_is_reported_as_unassessable():
    # Only low flow measured early, only high flow late -> a shift is not
    # separable from the curve shape.
    n = 50
    dates = pd.date_range("2018-01-01", periods=n, freq="20D")
    h = np.linspace(0.3, 2.2, n)               # rises monotonically with date
    q = 5.0 * (h - 0.15) ** 2.0 * np.exp(np.random.default_rng(0).normal(0, 0.03, n))
    df = pd.DataFrame({"date": dates, "stage_m": h, "discharge_cms": q, "is_valid": True})

    drift = fit_rating_curve(df, segments=1, random_state=0)["drift"]
    assert drift["flag"] == "unassessable"
    assert abs(drift["stage_time_corr"]) > 0.9
    assert "cannot be separated" in drift["message"]
