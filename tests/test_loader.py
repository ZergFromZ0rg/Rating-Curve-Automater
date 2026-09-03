import numpy as np
import pandas as pd

from rating_curve_automater.schema import DATE, DISCHARGE_CMS, STAGE_M
from rating_curve_automater.loader import load_measurements
from rating_curve_automater.workflow import RatingCurveWorkflow


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


def test_ambiguous_stage_message_points_at_the_override(tmp_path):
    path = tmp_path / "twostage.xlsx"
    _write(path, {"S": pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=5),
        "Stage (m)": [0.3, 0.4, 0.5, 0.6, 0.7],
        "Water level (m)": [10.3, 10.4, 10.5, 10.6, 10.7],
        "Q (m3/s)": [0.05, 0.1, 0.2, 0.3, 0.45],
    })})

    _, report = load_measurements(path)

    stage_msgs = [m for m in report.messages if m.startswith("stage_m:")]
    assert stage_msgs and "--stage-column" in stage_msgs[0]


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


def test_two_row_header_is_combined(tmp_path):
    path = tmp_path / "tworow.xlsx"
    n = 20
    rows = [["Date", "Stage", "Discharge"], ["", "(m)", "(m3/s)"]]
    for i in range(n):
        rows.append([f"2021-01-{i + 1:02d}", 0.3 + 0.03 * i, 0.1 + 0.05 * i])
    pd.DataFrame(rows).to_excel(path, sheet_name="Sheet1", index=False, header=False)

    canonical, report = load_measurements(path)

    assert report.ok
    assert report.two_row_header
    assert report.header_row == 0
    assert report.n_rows == n
    # unit came from the second header row and was converted
    assert report.units[STAGE_M].label == "m"


def test_section_title_row_is_not_mistaken_for_header(tmp_path):
    path = tmp_path / "titled.xlsx"
    rows = [
        ["River gauging record - site 12", None, None],
        [None, None, None],
        ["Date", "Gauge height (m)", "Flow (m3/s)"],
    ]
    for i in range(15):
        rows.append([f"2021-02-{i + 1:02d}", 0.4 + 0.02 * i, 0.2 + 0.03 * i])
    pd.DataFrame(rows).to_excel(path, sheet_name="Sheet1", index=False, header=False)

    _, report = load_measurements(path)
    assert report.header_row == 2
    assert report.ok


def test_ambiguous_sheets_lower_confidence(tmp_path):
    path = tmp_path / "twins.xlsx"
    frame = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=8, freq="MS"),
        "Stage (m)": np.linspace(0.3, 1.0, 8),
        "Discharge (m3/s)": np.linspace(0.1, 0.9, 8),
    })
    _write(path, {"2019": frame, "2020": frame})

    _, report = load_measurements(path)
    assert not report.sheet_confident
    assert report.needs_review


def test_explicit_header_row_override(tmp_path):
    path = tmp_path / "override.xlsx"
    data = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=6, freq="MS"),
        "Stage (m)": np.linspace(0.3, 1.0, 6),
        "Discharge (m3/s)": np.linspace(0.1, 0.9, 6),
    })
    _write(path, {"Data": data}, startrow={"Data": 4})

    _, report = load_measurements(path, header_row=4)
    assert report.header_row == 4
    assert report.header_confident
    assert report.ok


def test_messy_values_and_footer_row(tmp_path):
    path = tmp_path / "messy.xlsx"
    rows = [
        {"Date": "01/02/2021", "Stage (m)": "0,45", "Discharge (m3/s)": "<0.01", "Site": "A"},
        {"Date": "15/02/2021", "Stage (m)": "1,20", "Discharge (m3/s)": "1 234,5", "Site": "A"},
        {"Date": "20/02/2021", "Stage (m)": "n/a", "Discharge (m3/s)": "2.0", "Site": "A"},
        {"Date": "Total", "Stage (m)": "", "Discharge (m3/s)": "", "Site": ""},
    ]
    pd.DataFrame(rows).to_excel(path, sheet_name="S", index=False)

    canonical, report = load_measurements(path)

    assert report.n_rows == 3  # footer dropped
    assert canonical[STAGE_M].iloc[0] == 0.45           # decimal comma
    assert canonical[DISCHARGE_CMS].iloc[1] == 1234.5   # thousands + decimal comma
    assert canonical[DATE].iloc[0].month == 2 and canonical[DATE].iloc[0].day == 1
    assert any("censored" in m for m in report.messages)


def test_separate_date_and_time_columns(tmp_path):
    path = tmp_path / "datetime.xlsx"
    n = 12
    pd.DataFrame({
        "Date": pd.date_range("2021-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "Time": ["08:30"] * n,
        "Stage (m)": np.linspace(0.3, 1.0, n),
        "Discharge (m3/s)": np.linspace(0.1, 0.8, n),
    }).to_excel(path, sheet_name="S", index=False)

    canonical, report = load_measurements(path)
    assert "time" not in canonical.columns
    assert canonical["date"].iloc[0].hour == 8 and canonical["date"].iloc[0].minute == 30
    assert any("date and time" in m for m in report.messages)


def test_multi_site_load_and_per_site_fit(tmp_path):
    path = tmp_path / "multisite.xlsx"
    frames = []
    for site, (a, b) in {"Upper": (1.0, 1.6), "Lower": (2.0, 1.9)}.items():
        h = np.linspace(0.3, 1.4, 40)
        frames.append(pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=40, freq="W"),
            "Stage (m)": h,
            "Discharge (m3/s)": a * (h - 0.1) ** b,
            "Site": site,
        }))
    pd.concat(frames).to_excel(path, sheet_name="S", index=False)

    wf = RatingCurveWorkflow()
    validation = wf.load_and_validate(path)
    assert validation.is_multi_site
    assert validation.sites == ["Lower", "Upper"]

    upper = wf.run_fit(site="Upper")
    assert upper.site == "Upper"
    assert upper.params["n_points"] == 40
    assert 1.4 < upper.params["b"] < 1.8

    out = wf.export_report(tmp_path / "upper.xlsx")
    summary = pd.read_excel(out, sheet_name="Summary")
    assert (summary["Metric"] == "site").any()


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
