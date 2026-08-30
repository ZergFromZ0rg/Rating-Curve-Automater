from pathlib import Path

import pandas as pd
import pytest

from src.workflow import RatingCurveWorkflow

DATASET = Path(__file__).resolve().parents[1] / "10_year_single_site_rating_curve_data.xlsx"


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

    out = wf.export_report(tmp_path / "report.xlsx", uncertainty_threshold=0.3)
    assert out.exists()
    assert "Summary" in pd.ExcelFile(out).sheet_names


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
