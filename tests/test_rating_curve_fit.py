import pandas as pd
import numpy as np

import pytest

from src.rating_curve_fitting import (
    estimate_h0,
    fit_rating_curve,
    predict_discharge,
    select_valid_measurements,
)


def _synthetic_curve(h0=0.18, a_true=1.18, b_true=1.72, n=80):
    h = np.linspace(h0 + 0.04, 1.1, n)
    q = a_true * (h - h0) ** b_true
    q_noisy = q * (1 + np.linspace(-0.015, 0.02, len(h)))
    return pd.DataFrame({
        "Stage Above Bed (m)": h,
        "Measured Discharge Q (m³/s)": q_noisy,
        "is_valid": True,
    })


def test_fit_rating_curve_recovers_expected_parameters():
    h0 = 0.18
    a_true = 1.18
    b_true = 1.72

    h = np.linspace(0.22, 1.1, 80)
    q = a_true * (h - h0) ** b_true
    noise = np.linspace(-0.015, 0.02, len(h))
    q_noisy = q * (1 + noise)

    df = pd.DataFrame({
        "Stage Above Bed (m)": h,
        "Measured Discharge Q (m³/s)": q_noisy,
        "is_valid": True,
    })

    result = fit_rating_curve(df, h0=h0)

    assert result["a"] > 0.8
    assert result["a"] < 1.6
    assert result["b"] > 1.4
    assert result["b"] < 2.1
    assert result["r_squared"] > 0.9


def test_estimate_h0_recovers_true_offset():
    df = _synthetic_curve(h0=0.30)
    stage = df["Stage Above Bed (m)"].to_numpy()
    discharge = df["Measured Discharge Q (m³/s)"].to_numpy()

    assert abs(estimate_h0(stage, discharge) - 0.30) < 0.05


def test_fit_rating_curve_estimates_h0_when_missing():
    df = _synthetic_curve(h0=0.30)

    result = fit_rating_curve(df)

    assert result["h0_estimated"] is True
    assert abs(result["h0"] - 0.30) < 0.05
    assert result["r_squared"] > 0.98
    assert result["n_points"] == len(df)


def test_user_supplied_h0_is_not_flagged_as_estimated():
    df = _synthetic_curve()

    result = fit_rating_curve(df, h0=0.18)

    assert result["h0"] == 0.18
    assert result["h0_estimated"] is False


def _segmented_frame(h0=0.18, break_stage=0.60):
    low_stage = np.linspace(h0 + 0.05, break_stage, 30)
    low_q = 0.5 * (low_stage - h0) ** 1.30
    q_at_break = 0.5 * (break_stage - h0) ** 1.30
    a_high = q_at_break / (break_stage - h0) ** 2.40
    high_stage = np.linspace(break_stage, 1.3, 30)
    high_q = a_high * (high_stage - h0) ** 2.40
    return pd.DataFrame({
        "Stage Above Bed (m)": np.concatenate([low_stage, high_stage]),
        "Measured Discharge Q (m³/s)": np.concatenate([low_q, high_q]),
        "is_valid": True,
    })


def test_fit_two_segments_recovers_breakpoint_and_exponents():
    df = _segmented_frame(break_stage=0.60)

    fit = fit_rating_curve(df, h0=0.18, segments=2)

    assert fit["is_segmented"] is True
    assert 0.45 < fit["breakpoint"] < 0.80
    assert len(fit["segments"]) == 2
    assert 1.0 < fit["segments"][0]["b"] < 1.7
    assert 2.0 < fit["segments"][1]["b"] < 2.9
    assert fit["r_squared"] > 0.99


def test_predict_discharge_matches_both_model_kinds():
    df = _segmented_frame()
    stage = df["Stage Above Bed (m)"].to_numpy()
    observed = df["Measured Discharge Q (m³/s)"].to_numpy()

    single = fit_rating_curve(df, h0=0.18, segments=1)
    segmented = fit_rating_curve(df, h0=0.18, segments=2)

    assert np.corrcoef(predict_discharge(single, stage), observed)[0, 1] > 0.98
    # the segmented model tracks the kinked data more closely
    seg_err = np.abs(predict_discharge(segmented, stage) - observed).mean()
    single_err = np.abs(predict_discharge(single, stage) - observed).mean()
    assert seg_err < single_err


def test_invalid_segment_count_raises():
    with pytest.raises(ValueError):
        fit_rating_curve(_segmented_frame(), segments=3)


def test_two_segments_with_too_few_points_raises():
    df = _synthetic_curve(n=6)
    with pytest.raises(ValueError):
        fit_rating_curve(df, h0=0.18, segments=2)


def test_select_valid_measurements_applies_flag_and_dropna():
    df = pd.DataFrame({
        "Stage Above Bed (m)": [0.4, 0.5, None, 0.7],
        "Measured Discharge Q (m³/s)": [0.1, 0.2, 0.3, 0.4],
        "is_valid": [True, False, True, True],
    })

    kept = select_valid_measurements(df)

    assert len(kept) == 2
    assert kept["Stage Above Bed (m)"].tolist() == [0.4, 0.7]
