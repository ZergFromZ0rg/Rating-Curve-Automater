import pandas as pd
import numpy as np

import pytest

from rating_curve_automater.rating_curve_fitting import (
    ImplausibleRatingCurve,
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


def test_estimate_h0_does_not_collapse_to_zero_on_a_compound_channel():
    # Two controls (b 1.4 -> 2.6) breaking at H = 0.9, true h0 = 0.15. The old
    # R²-maximising search railed to ~0 here because a, b and h0 trade off.
    h0_true, brk = 0.15, 0.90
    rng = np.random.default_rng(1)
    stage = np.sort(rng.uniform(0.30, 2.2, 60))
    x = stage - h0_true
    q_low = 6.0 * x ** 1.4
    a_high = (6.0 * (brk - h0_true) ** 1.4) / (brk - h0_true) ** 2.6
    discharge = np.where(stage < brk, q_low, a_high * x ** 2.6)
    discharge *= np.exp(rng.normal(0, 0.04, stage.size))

    est = estimate_h0(stage, discharge)
    assert 0.05 < est < 0.28  # near the truth, nowhere near a collapsed 0


def test_fit_rating_curve_reports_h0_diagnostics_and_flags_a_railed_estimate():
    df = _synthetic_curve(h0=0.30)
    fit = fit_rating_curve(df)
    assert "h0_diagnostics" in fit
    assert fit["h0_diagnostics"]["railed"] is False

    # True point of zero flow well below the datum (h0 = -0.8) sits past the
    # search's lower bound, so the estimate rails there -> flagged + warned.
    h = np.linspace(0.5, 4.0, 40)
    deep = pd.DataFrame({
        "Stage Above Bed (m)": h,
        "Measured Discharge Q (m³/s)": 2.0 * (h + 0.8) ** 1.9,
        "is_valid": True,
    })
    fit2 = fit_rating_curve(deep)
    assert fit2["h0_diagnostics"]["railed"] is True
    assert any("weakly identified" in w for w in fit2["warnings"])


def test_single_power_law_on_a_kinked_channel_warns_to_use_segments():
    df = _segmented_frame(break_stage=0.60)
    fit = fit_rating_curve(df, segments=1)
    assert any("log-log space" in w and "segments" in w for w in fit["warnings"])


def test_leave_one_out_error_is_honest_about_out_of_sample_accuracy():
    from rating_curve_automater.rating_curve_fitting import leave_one_out_error

    rng = np.random.default_rng(4)
    h = np.sort(rng.uniform(0.3, 1.6, 40))
    q = 1.2 * (h - 0.15) ** 1.8 * np.exp(rng.normal(0, 0.05, 40))
    q[10] *= 1.5  # one blunder
    df = pd.DataFrame({"Stage Above Bed (m)": h, "Measured Discharge Q (m³/s)": q, "is_valid": True})

    loo = leave_one_out_error(df, segments=1)
    fit = fit_rating_curve(df, segments=1)

    assert loo is not None and loo["n"] >= 8
    assert set(loo) == {"n", "rmspe_pct", "bias_pct", "mae_pct", "p95_abs_pct"}
    # out-of-sample error is real and at least as large as the in-sample miss
    assert loo["rmspe_pct"] > 0
    in_sample = 100 * np.sqrt(1 - fit["r_squared"])  # rough
    assert loo["rmspe_pct"] >= in_sample * 0.5

    # too few points -> None
    assert leave_one_out_error(df.head(5)) is None


def test_auto_fit_on_a_compound_channel_recovers_h0_and_the_break():
    # Realistic gauging record: low-flow visits dominate. True h0 = 0.15, control
    # break at H = 0.90. The estimated h0 must not collapse toward 0, and the
    # auto fit must pick up the second control.
    h0_true, brk = 0.15, 0.90
    rng = np.random.default_rng(3)
    stage = np.sort(0.20 + rng.beta(1.6, 2.4, 70) * 2.1)
    x = np.maximum(stage - h0_true, 1e-9)
    a_high = (8.0 * (brk - h0_true) ** 2.1) / (brk - h0_true) ** 2.7
    discharge = np.where(stage < brk, 8.0 * x ** 2.1, a_high * x ** 2.7)
    discharge *= np.exp(rng.normal(0, 0.06, stage.size))
    df = pd.DataFrame({"Stage Above Bed (m)": stage,
                       "Measured Discharge Q (m³/s)": discharge, "is_valid": True})

    fit = fit_rating_curve(df, segments="auto", random_state=0)
    assert fit["h0"] > 0.05                       # not collapsed
    assert abs(fit["h0"] - h0_true) < 0.12
    assert fit["is_segmented"] and fit["n_segments"] == 2
    assert abs(fit["breakpoints"][0] - brk) < 0.25


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
    assert fit["n_segments"] == 2
    assert 0.45 < fit["breakpoint"] < 0.80
    assert fit["breakpoints"] == [fit["breakpoint"]]
    assert len(fit["segments"]) == 2
    assert 1.0 < fit["segments"][0]["b"] < 1.7
    assert 2.0 < fit["segments"][1]["b"] < 2.9
    assert fit["r_squared"] > 0.99


def test_continuous_piecewise_curve_has_no_jump_at_the_breakpoint():
    df = _segmented_frame(break_stage=0.60)
    fit = fit_rating_curve(df, h0=0.18, segments=2)

    bp = fit["breakpoint"]
    left = predict_discharge(fit, bp - 1e-4)[0]
    right = predict_discharge(fit, bp + 1e-4)[0]
    assert abs(left - right) / right < 1e-3


def test_three_segments_can_be_forced():
    df = _segmented_frame(break_stage=0.60)
    fit = fit_rating_curve(df, h0=0.18, segments=3)

    assert fit["n_segments"] == 3
    assert len(fit["segments"]) == 3
    assert len(fit["breakpoints"]) == 2
    assert fit["breakpoints"] == sorted(fit["breakpoints"])


def test_auto_segments_picks_two_for_kinked_data_and_one_for_a_clean_line():
    rng = np.random.default_rng(0)
    df = _segmented_frame(break_stage=0.60)
    df["Measured Discharge Q (m³/s)"] *= np.exp(rng.normal(0, 0.05, len(df)))

    kinked = fit_rating_curve(df, h0=0.18, segments="auto", random_state=0)
    assert kinked["is_segmented"] is True
    assert kinked["n_segments"] == 2
    assert kinked["segment_selection"] == "auto"

    straight_df = _synthetic_curve()
    straight_df["Measured Discharge Q (m³/s)"] *= np.exp(rng.normal(0, 0.03, len(straight_df)))
    straight = fit_rating_curve(straight_df, h0=0.18, segments="auto", random_state=0)
    assert straight["is_segmented"] is False
    assert straight["n_segments"] == 1


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


@pytest.mark.parametrize("bad", [0, -1, "banana", 2.5])
def test_invalid_segment_argument_raises(bad):
    with pytest.raises(ValueError):
        fit_rating_curve(_segmented_frame(), segments=bad)


def test_two_segments_with_too_few_points_raises():
    df = _synthetic_curve(n=6)
    with pytest.raises(ValueError):
        fit_rating_curve(df, h0=0.18, segments=2)


def test_plausible_fit_has_no_warnings():
    result = fit_rating_curve(_synthetic_curve(), h0=0.18)
    assert result["is_plausible"] is True
    assert result["warnings"] == []


def test_inverse_relationship_is_flagged_and_can_raise():
    # discharge falls as stage rises -> not a rating curve
    df = pd.DataFrame({
        "Stage Above Bed (m)": [0.58, 0.62, 0.87, 0.95, 1.05],
        "Measured Discharge Q (m³/s)": [0.52, 0.13, 0.05, 0.04, 0.03],
        "is_valid": True,
    })

    fit = fit_rating_curve(df, h0=0.4)
    assert fit["is_plausible"] is False
    assert any("b =" in w and "≤ 0" in w for w in fit["warnings"])

    with pytest.raises(ImplausibleRatingCurve):
        fit_rating_curve(df, h0=0.4, strict=True)


def test_fixed_exponent_fits_only_the_coefficient():
    # A clean b=2 curve: imposing b=2 should recover a and leave b exactly 2.
    h = np.linspace(0.25, 1.2, 40)
    q = 0.7 * (h - 0.18) ** 2.0
    df = pd.DataFrame({"stage_m": h, "discharge_cms": q, "is_valid": True})

    fit = fit_rating_curve(df, h0=0.18, fixed_b=2.0)

    assert fit["b"] == 2.0
    assert fit["b_fixed"] is True
    assert fit["a"] == pytest.approx(0.7, rel=1e-6)
    assert fit["r_squared"] > 0.999


def test_fixed_exponent_rescues_a_record_that_cannot_identify_b():
    # Scattered low-flow cloud: a free fit is rejected (b <= 0 / uncorrelated),
    # but imposing the control exponent yields a usable, if provisional, curve.
    rng = np.random.default_rng(0)
    h = rng.uniform(0.05, 0.5, 30)
    q = rng.uniform(0.02, 0.13, 30)
    df = pd.DataFrame({"stage_m": h, "discharge_cms": q, "is_valid": True})

    free = fit_rating_curve(df, h0=0.0)
    assert free["is_plausible"] is False
    assert any(w.startswith("→") for w in free["warnings"])  # actionable guidance

    imposed = fit_rating_curve(df, h0=0.0, fixed_b=2.0, n_bootstrap=200, random_state=0)
    assert imposed["b"] == 2.0
    assert imposed["is_plausible"] is True          # not "not a rating curve"
    assert not any("≤ 0" in w for w in imposed["warnings"])
    assert imposed["bands"]["b_ci"] is None         # b has no CI when it is imposed
    assert imposed["bands"]["a_ci"] is not None


@pytest.mark.parametrize("bad_kwargs", [
    {"fixed_b": 0.0},
    {"fixed_b": -1.0},
    {"fixed_b": 2.0, "segments": 2},
    {"fixed_b": 2.0, "segments": "auto"},
    {"fixed_b": 2.0, "method": "bayesian"},
])
def test_fixed_exponent_rejects_unsupported_combinations(bad_kwargs):
    df = _synthetic_curve()
    with pytest.raises(ValueError):
        fit_rating_curve(df, h0=0.18, n_bootstrap=0, **bad_kwargs)


def test_too_few_points_is_a_non_critical_warning():
    df = pd.DataFrame({
        "Stage Above Bed (m)": [0.25, 0.6, 1.0],
        "Measured Discharge Q (m³/s)": [0.05, 0.3, 1.0],
        "is_valid": True,
    })
    fit = fit_rating_curve(df, h0=0.18)
    assert fit["is_plausible"] is True  # not critical
    assert any("only 3 point" in w for w in fit["warnings"])


def _clean_curve(h0=0.18, a_true=1.2, b_true=1.7, n=60):
    h = np.linspace(h0 + 0.05, 1.3, n)
    q = a_true * (h - h0) ** b_true
    return h, q


def test_uniform_uncertainty_column_does_not_change_the_fit():
    h, q = _clean_curve()
    base = pd.DataFrame({"stage_m": h, "discharge_cms": q, "is_valid": True})
    with_col = base.assign(discharge_uncertainty=8.0)  # every point ±8 %

    a0 = fit_rating_curve(base, h0=0.18)
    a1 = fit_rating_curve(with_col, h0=0.18)

    assert a1["uncertainty_source"] == "column"
    assert a1["weighted"] is False
    assert a1["a"] == pytest.approx(a0["a"], rel=1e-9)
    assert a1["b"] == pytest.approx(a0["b"], rel=1e-9)


def test_default_uncertainty_pct_has_no_effect_without_a_column():
    h, q = _clean_curve()
    df = pd.DataFrame({"stage_m": h, "discharge_cms": q, "is_valid": True})

    low = fit_rating_curve(df, h0=0.18, discharge_uncertainty_pct=3)
    high = fit_rating_curve(df, h0=0.18, discharge_uncertainty_pct=40)

    assert low["uncertainty_source"] == "default"
    assert low["a"] == pytest.approx(high["a"], rel=1e-9)
    assert low["b"] == pytest.approx(high["b"], rel=1e-9)


def test_per_point_uncertainty_downweights_noisy_gaugings():
    h, q = _clean_curve(b_true=1.7, n=60)
    q_noisy = q.copy()
    # Wreck the three highest-stage gaugings.
    q_noisy[-3:] *= [0.45, 0.4, 0.35]

    unc = np.full(len(h), 4.0)
    unc[-3:] = 60.0  # ...but flag them as very uncertain

    df = pd.DataFrame({"stage_m": h, "discharge_cms": q_noisy, "discharge_uncertainty": unc, "is_valid": True})

    unweighted = fit_rating_curve(df.drop(columns="discharge_uncertainty"), h0=0.18)
    weighted = fit_rating_curve(df, h0=0.18)

    assert weighted["weighted"] is True
    # The true exponent is 1.7; down-weighting the bad points recovers it better.
    assert abs(weighted["b"] - 1.7) < abs(unweighted["b"] - 1.7)
    assert weighted["b"] == pytest.approx(1.7, abs=0.1)


def test_per_point_uncertainty_also_weights_segmented_fit():
    df = _segmented_frame(break_stage=0.60)
    df["discharge_uncertainty"] = np.linspace(3.0, 9.0, len(df))

    fit = fit_rating_curve(df, h0=0.18, segments=2)

    assert fit["is_segmented"] is True
    assert fit["weighted"] is True
    assert 0.45 < fit["breakpoint"] < 0.80


def test_weighted_r_squared_is_used_for_the_poor_fit_check():
    h, q = _clean_curve(b_true=1.7, n=50)
    q[-8:] *= 0.25  # badly ruin the top gaugings
    unc = np.full(len(h), 3.0)
    unc[-8:] = 90.0
    df = pd.DataFrame({"stage_m": h, "discharge_cms": q, "discharge_uncertainty": unc, "is_valid": True})

    fit = fit_rating_curve(df, h0=0.18)

    assert fit["weighted"] is True
    assert fit["r_squared"] < 0.5                          # unweighted R² is dragged down...
    assert fit["r_squared_weighted"] > fit["r_squared"]    # ...but the weighted fit is good
    assert fit["r_squared_weighted"] > 0.9
    assert not any("poor fit" in w for w in fit["warnings"])


def test_n_bootstrap_populates_bands_on_the_fit():
    df = _synthetic_curve()
    fit = fit_rating_curve(df, h0=0.18, n_bootstrap=200, random_state=0)
    assert fit["bands"] is not None
    assert len(fit["bands"]["stage"]) == len(fit["bands"]["ci_lower"])


def test_select_valid_measurements_applies_flag_and_dropna():
    df = pd.DataFrame({
        "Stage Above Bed (m)": [0.4, 0.5, None, 0.7],
        "Measured Discharge Q (m³/s)": [0.1, 0.2, 0.3, 0.4],
        "is_valid": [True, False, True, True],
    })

    kept = select_valid_measurements(df)

    assert len(kept) == 2
    assert kept["stage_m"].tolist() == [0.4, 0.7]
