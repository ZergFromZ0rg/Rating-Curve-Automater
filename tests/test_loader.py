import numpy as np
import pandas as pd
import pytest

from src.schema import DATE, DISCHARGE_CMS, STAGE_M
from src.loader import load_measurements
from src.workflow import RatingCurveWorkflow


def _write(path, frames: dict[str, pd.DataFrame], startrow: dict[str, int] | None = None):
    startrow = startrow or {}
    with pd.ExcelWriter(path) as writer:
        for sheet, frame in frames.items():
            frame.to_excel(writer, sheet_name=sheet, index=False, startrow=startrow.get(sheet, 0))


def test_load_varied_headers_and_units(tmp_path):
    path = tmp_path / "foreign.xlsx"
    n = 40
    stage_ft = np.linspace(1.0, 5.0, n)
    q_cfs = 3.0 * (stage_ft - 0.5) ** 1.7
    _write(path, {"Gaugings": pd.DataFrame({
        "Sample Date": pd.date_range("2019-01-01", periods=n, freq="W"),
        "Gauge Height (ft)": stage_ft,
        "Streamflow (cfs)": q_cfs,
        "Grade": ["Good"] * n,
    })})

    canonical, report = load_measurements(path)

    assert report.ok
    assert list(canonical.columns)[:3] == [DATE, STAGE_M, DISCHARGE_CMS]
    assert report.units[STAGE_M].detected and report.units[STAGE_M].label == "ft"
    # ft -> m conversion applied
    assert abs(canonical[STAGE_M].iloc[0] - 1.0 * 0.3048) < 1e-6
    assert canonical[DISCHARGE_CMS].iloc[-1] < q_cfs[-1]


def test_auto_selects_data_sheet_and_header_row(tmp_path):
    path = tmp_path / "with_meta.xlsx"
    data = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=10, freq="MS"),
        "Stage (m)": np.linspace(0.3, 1.2, 10),
        "Discharge (m3/s)": np.linspace(0.1, 1.5, 10),
    })
    _write(
        path,
        {
            "Notes": pd.DataFrame({"About": ["station metadata", "line 2"]}),
            "Data": data,
        },
        startrow={"Data": 3},  # 3 blank/title rows above the header
    )

    canonical, report = load_measurements(path)

    assert report.sheet_name == "Data"
    assert report.header_row == 3
    assert report.n_rows == 10
    assert report.ok


def test_missing_required_column_reports_not_raises(tmp_path):
    path = tmp_path / "incomplete.xlsx"
    _write(path, {"Sheet1": pd.DataFrame({"Date": ["2020-01-01"], "Air Temp": [12.0]})})

    _, report = load_measurements(path)
    assert not report.ok
    assert STAGE_M in report.mapping.unresolved_required


def test_column_overrides(tmp_path):
    path = tmp_path / "cryptic.xlsx"
    _write(path, {"S": pd.DataFrame({
        "d": pd.date_range("2020-01-01", periods=5),
        "x1": [0.3, 0.4, 0.5, 0.6, 0.7],
        "x2": [0.05, 0.1, 0.2, 0.3, 0.45],
    })})

    canonical, report = load_measurements(
        path, column_overrides={DATE: "d", STAGE_M: "x1", DISCHARGE_CMS: "x2"}
    )
    assert report.ok
    assert canonical[STAGE_M].tolist() == [0.3, 0.4, 0.5, 0.6, 0.7]


def test_csv_input(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=6, freq="MS"),
        "Gauge Height (m)": np.linspace(0.3, 1.0, 6),
        "Flow (m3/s)": np.linspace(0.1, 0.9, 6),
    }).to_csv(path, index=False)

    canonical, report = load_measurements(path)
    assert report.ok
    assert report.sheet_name is None
    assert len(canonical) == 6


def test_workflow_end_to_end_on_foreign_sheet(tmp_path):
    path = tmp_path / "foreign2.xlsx"
    n = 50
    h = np.linspace(0.25, 1.4, n)
    q = 1.1 * (h - 0.15) ** 1.8
    _write(path, {"gaugings": pd.DataFrame({
        "Gauging Date": pd.date_range("2018-01-01", periods=n, freq="W"),
        "Stage height (m)": h,
        "Gauged discharge (m3/s)": q,
    })})

    wf = RatingCurveWorkflow()
    validation = wf.load_and_validate(path)
    assert validation.valid_count == n
    outcome = wf.run_fit()
    assert outcome.params["r_squared"] > 0.99
    out = wf.export_report(tmp_path / "r.xlsx")
    assert out.exists()
