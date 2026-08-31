import numpy as np
import pandas as pd
import pytest

from rating_curve_automater.manning import (
    manning_q,
    manning_sanity_check,
    read_cross_section,
    section_hydraulics,
)
from rating_curve_automater.rating_curve_fitting import fit_rating_curve

# A rectangular channel 4 m wide, vertical walls 3 m high, on a wide shelf.
_RECT_X = np.array([-20.0, -2.0, -2.0, 2.0, 2.0, 20.0])
_RECT_Z = np.array([3.0, 3.0, 0.0, 0.0, 3.0, 3.0])


def test_section_hydraulics_rectangular_channel():
    # water 1 m deep in a 4 m rectangle: A = 4, P = 4 + 2*1 = 6, T = 4
    area, perim, width = section_hydraulics(_RECT_X, _RECT_Z, wse=1.0)
    assert area == pytest.approx(4.0)
    assert perim == pytest.approx(6.0)
    assert width == pytest.approx(4.0)


def test_section_hydraulics_multiple_channels():
    # two 2 m notches separated by a 2 m island above the water line
    x = np.array([-5.0, -3.0, -3.0, -1.0, -1.0, 1.0, 1.0, 3.0, 3.0, 5.0])
    z = np.array([2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0])
    area, perim, width = section_hydraulics(x, z, wse=1.0)
    assert area == pytest.approx(4.0)          # 2 notches * (2 m wide * 1 m deep)
    assert width == pytest.approx(4.0)


def test_manning_q_matches_hand_calculation():
    # A=4, P=6, R=2/3, S=0.001, n=0.03
    q = manning_q(4.0, 6.0, slope=0.001, n=0.03)
    expected = (1 / 0.03) * 4.0 * (2 / 3) ** (2 / 3) * 0.001 ** 0.5
    assert q == pytest.approx(expected)
    assert manning_q(0.0, 6.0, 0.001, 0.03) == 0.0


def test_read_cross_section_by_column_name(tmp_path):
    p = tmp_path / "xs.csv"
    pd.DataFrame({"Station (m)": [5, 0, 10, 3], "Bed elevation": [2.0, 0.0, 2.0, 0.5]}).to_csv(p, index=False)
    off, elev = read_cross_section(p)
    assert list(off) == [0, 3, 5, 10]          # sorted by offset
    assert elev[0] == 0.0


def _notch_to_wide_channel():
    """Gaugings live in a narrow low-flow notch; above it the channel is 10x wider."""
    x = np.array([-25, -5.0, -5.0, -0.4, -0.4, 0.4, 0.4, 5.0, 5.0, 25.0])
    z = np.array([4.0, 4.0, 0.6, 0.6, 0.0, 0.0, 0.6, 0.6, 4.0, 4.0])
    slope, n = 0.0015, 0.032
    true_q = np.array([
        manning_q(*section_hydraulics(x, z, float(h))[:2], slope, n)
        for h in np.linspace(0.05, 0.55, 25)
    ])
    df = pd.DataFrame({"stage_m": np.linspace(0.05, 0.55, 25),
                       "discharge_cms": true_q * np.exp(np.random.default_rng(0).normal(0, 0.03, 25)),
                       "is_valid": True})
    return x, z, slope, n, df


def test_calibrated_n_recovers_the_true_roughness():
    x, z, slope, n_true, df = _notch_to_wide_channel()
    fit = fit_rating_curve(df, segments="auto", random_state=0)
    res = manning_sanity_check(fit, x, z, slope, extrapolate_to=0.5)  # stay in the notch
    assert res["n_calibrated"] == pytest.approx(n_true, abs=0.005)
    assert res["flag"] == "ok"


def test_flags_a_power_law_that_cannot_see_the_wider_channel():
    x, z, slope, _, df = _notch_to_wide_channel()
    fit = fit_rating_curve(df, segments="auto", random_state=0)
    res = manning_sanity_check(fit, x, z, slope, extrapolate_to=1.5)
    assert res["flag"] == "diverges"
    assert res["max_abs_pct_diff_extrapolated"] > 35
    assert "not supported by the channel geometry" in res["message"]
    # in-range the two curves agree
    assert res["max_abs_pct_diff_gauged"] < 10


def test_implausible_n_is_flagged():
    x, z, slope, _, df = _notch_to_wide_channel()
    fit = fit_rating_curve(df, segments="auto", random_state=0)
    res = manning_sanity_check(fit, x, z, slope * 100)   # absurd slope -> absurd n
    assert res["flag"] == "implausible-n"
    assert res["n_used"] > 0.20


def test_unusable_inputs_do_not_raise():
    df = pd.DataFrame({"stage_m": np.linspace(0.2, 1.0, 12),
                       "discharge_cms": np.linspace(0.5, 8, 12), "is_valid": True})
    fit = fit_rating_curve(df, segments=1, random_state=0)
    bad = manning_sanity_check(fit, [0, 1, 2, 3], [1, 1, 1, 1], slope=-1.0)
    assert bad["flag"] == "unusable"


def test_workflow_and_report_include_the_manning_check(tmp_path):
    from rating_curve_automater import RatingCurveWorkflow

    x, z, slope, _, df = _notch_to_wide_channel()
    df["date"] = pd.date_range("2022-01-01", periods=len(df), freq="9D")
    g = tmp_path / "g.csv"
    df[["date", "stage_m", "discharge_cms"]].to_csv(g, index=False)
    xs = tmp_path / "xs.csv"
    pd.DataFrame({"offset": x, "elevation": z}).to_csv(xs, index=False)

    wf = RatingCurveWorkflow()
    wf.load_and_validate(str(g))
    wf.run_fit(segments="auto")
    res = wf.manning_check(str(xs), slope)
    assert res["flag"] in ("diverges", "check")
    assert wf.fit_params["manning"] is res

    out = wf.export_report(tmp_path / "r.xlsx")
    import openpyxl

    wb = openpyxl.load_workbook(out)
    assert "Manning Check" in wb.sheetnames
    assert wb["Manning Check"]["H2"].value == res["flag"]
