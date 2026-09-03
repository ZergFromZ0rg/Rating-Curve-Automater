from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rating_curve_automater.workflow import RatingCurveWorkflow

DATASET = Path(__file__).resolve().parents[1] / "rating_curve_automater" / "data" / "10_year_single_site_rating_curve_data.xlsx"


@pytest.fixture
def workbook(tmp_path):
    path = tmp_path / "measurements.xlsx"
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=6, freq="MS"),
            "Stage Above Bed (m)": [0.30, 0.45, 0.60, 0.80, 1.00, -0.20],
            "Measured Discharge Q (m³/s)": [0.05, 0.15, 0.32, 0.70, 1.30, 0.10],
            "Quality": ["Good", "Good", "Fair", "Good", "Good", "Good"],
            "Field Notes": ["Normal", "Normal", "Vegetation / possible backwater", "Normal", "Normal", "Normal"],
        }
    )
    df.to_excel(path, sheet_name="Measurements", index=False)
    return path


def test_workflow_validate_fit_export(workbook, tmp_path):
    wf = RatingCurveWorkflow()

    validation = wf.load_and_validate(workbook)
    assert validation.invalid_count == 1
    assert validation.warning_count == 1
    assert validation.has_blocking_issues
    assert len(validation.flags) == 1
    assert "backwater" in validation.warnings[0].lower()

    outcome = wf.run_fit()
    assert outcome.params["n_points"] == 5  # the negative-stage row is dropped
    assert outcome.params["r_squared"] > 0.9

    assert outcome.bands is not None
    assert "CI b" in outcome.summary_line()

    out = wf.export_report(tmp_path / "report.xlsx", uncertainty_threshold=0.3)
    assert out.exists()
    sheets = pd.ExcelFile(out).sheet_names
    assert "Summary" in sheets
    assert "Rating Curve Band" in sheets
    assert "Rating Table" in sheets
    band = pd.read_excel(out, sheet_name="Rating Curve Band")
    assert (band.iloc[:, 2] <= band.iloc[:, 1] + 1e-6).all()  # confidence lower <= modeled

    rt = wf.rating_table(step=0.05)
    assert (rt["Stage (m)"].diff().dropna().round(6) == 0.05).all()
    assert rt["Discharge (m³/s)"].is_monotonic_increasing
    csv_out = wf.export_rating_table_csv(tmp_path / "rt.csv", step=0.05)
    assert csv_out.exists() and len(pd.read_csv(csv_out)) == len(rt)


def test_workflow_uses_a_mapped_uncertainty_column_to_weight_the_fit(tmp_path):
    path = tmp_path / "with_uncertainty.xlsx"
    n = 40
    h = pd.Series(range(n)) / (n - 1) * 1.1 + 0.25
    q = 1.2 * (h - 0.18) ** 1.7
    q_noisy = q.copy()
    q_noisy.iloc[-4:] *= 0.4  # ruin the top gaugings

    unc = [5.0] * n
    for i in range(1, 5):
        unc[-i] = 70.0  # ...but mark them very uncertain

    pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=n, freq="W"),
        "Stage Above Bed (m)": h,
        "Measured Discharge Q (m³/s)": q_noisy,
        "Discharge Uncertainty (%)": unc,
    }).to_excel(path, sheet_name="M", index=False)

    wf = RatingCurveWorkflow()
    validation = wf.load_and_validate(path)
    assert validation.load_report.mapping.fields["discharge_uncertainty"] == "Discharge Uncertainty (%)"

    outcome = wf.run_fit(h0=0.18)
    assert outcome.params["weighted"] is True
    assert outcome.params["b"] == pytest.approx(1.7, abs=0.12)
    assert "weighted" in outcome.summary_line()

    out = wf.export_report(tmp_path / "r.xlsx")
    summary = pd.read_excel(out, sheet_name="Summary")
    assert summary["Metric"].astype(str).str.contains("discharge uncertainty").any()


def test_workflow_flags_a_drifting_rating_and_writes_the_residual_sheet(tmp_path):
    rng = np.random.default_rng(0)
    n = 55
    days = np.sort(rng.uniform(0, 7 * 365, n))
    dates = pd.Timestamp("2016-01-01") + pd.to_timedelta(days, unit="D")
    h = rng.uniform(0.3, 1.4, n)
    q = 1.2 * (h - 0.15) ** 1.75 * np.exp(0.045 * days / 365.25) * np.exp(rng.normal(0, 0.04, n))
    path = tmp_path / "drift.xlsx"
    pd.DataFrame({
        "Date": dates,
        "Stage Above Bed (m)": h,
        "Measured Discharge Q (m³/s)": q,
    }).to_excel(path, sheet_name="M", index=False)

    wf = RatingCurveWorkflow()
    wf.load_and_validate(path)
    outcome = wf.run_fit(h0=0.15)

    assert outcome.drift["flag"] == "likely"
    assert "shift" in outcome.summary_line().lower()

    out = wf.export_report(tmp_path / "r.xlsx")
    sheets = pd.ExcelFile(out).sheet_names
    assert "Residuals Over Time" in sheets
    summary = pd.read_excel(out, sheet_name="Summary").astype(str)
    assert summary["Metric"].str.contains("temporal drift").any()


def test_rating_table_before_fit_raises(workbook):
    wf = RatingCurveWorkflow()
    wf.load_and_validate(workbook)
    with pytest.raises(RuntimeError):
        wf.rating_table()


def test_fit_before_validate_raises():
    wf = RatingCurveWorkflow()
    with pytest.raises(RuntimeError):
        wf.run_fit()


def test_export_before_fit_raises(workbook):
    wf = RatingCurveWorkflow()
    wf.load_and_validate(workbook)
    with pytest.raises(RuntimeError):
        wf.export_report("unused.xlsx")


@pytest.mark.skipif(not DATASET.exists(), reason="bundled dataset not present")
def test_workflow_on_bundled_dataset(tmp_path):
    wf = RatingCurveWorkflow()

    validation = wf.load_and_validate(DATASET)
    assert validation.valid_count == 120
    assert validation.warning_count == 12  # the 12 "Fair" points

    outcome = wf.run_fit()
    assert abs(outcome.params["h0"] - 0.18) < 0.05
    assert outcome.params["r_squared"] > 0.98


def test_workflow_run_fit_accepts_an_imposed_exponent(tmp_path):
    wf = RatingCurveWorkflow()
    wf.load_and_validate(DATASET)

    outcome = wf.run_fit(fixed_b=2.0)

    assert outcome.params["b"] == 2.0
    assert outcome.params["b_fixed"] is True
    assert "b imposed" in outcome.summary_line()
    out = wf.export_report(tmp_path / "imposed.xlsx")
    summary = pd.read_excel(out, sheet_name="Summary")
    assert (summary["Value"].astype(str) == "imposed by user (not fitted)").any()
