import re
import zipfile

import numpy as np
import pandas as pd

from rating_curve_automater.rating_curve_fitting import fit_rating_curve
from rating_curve_automater.rating_curve_report import build_observed_modeled_table, export_rating_curve_report


def test_build_observed_modeled_table_flags_uncertain_points():
    df = pd.DataFrame(
        {
            "Date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "Stage Above Bed (m)": [0.25, 0.50, 0.90],
            "Measured Discharge Q (m³/s)": [0.04, 0.18, 0.80],
        }
    )

    table = build_observed_modeled_table(df, a=1.18, b=1.72, h0=0.18)

    assert {"Date", "Stage Above Bed (m)", "Measured Discharge Q (m³/s)", "Modeled Discharge Q (m³/s)", "Residual", "Uncertainty Flag"}.issubset(table.columns)
    assert table["Uncertainty Flag"].isin(["Normal", "Uncertain"]).all()
    assert table["Uncertainty Flag"].str.contains("Uncertain").any()


def test_export_rating_curve_report_writes_excel(tmp_path):
    path = tmp_path / "report.xlsx"

    df = pd.DataFrame(
        {
            "Date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "Stage Above Bed (m)": [0.25, 0.50, 0.90],
            "Measured Discharge Q (m³/s)": [0.04, 0.18, 0.80],
        }
    )

    export_rating_curve_report(df, path, a=1.18, b=1.72, h0=0.18)

    assert path.exists()

    workbook = pd.ExcelFile(path)
    assert "Original Data" in workbook.sheet_names
    assert "Summary" in workbook.sheet_names
    assert "Plot" in workbook.sheet_names
    assert "Plot Data" in workbook.sheet_names

    with zipfile.ZipFile(path) as zf:
        chart_xml = zf.read("xl/charts/chart1.xml").decode("utf-8", errors="ignore")
        assert "numRef" in chart_xml
        assert "'Plot'!$B$2" in chart_xml
        assert "'Plot'!$C$2" in chart_xml


def test_report_chart_line_widths_are_integer_emu(tmp_path):
    # A fractional EMU width in the drawing XML makes Excel strip the chart on
    # open ("Removed Part: /xl/drawings/drawingN.xml") — every width must be int.
    rng = np.random.default_rng(0)
    h = np.linspace(0.25, 1.4, 40)
    q = 1.2 * (h - 0.15) ** 1.8 * (1 + rng.normal(0, 0.05, h.size))
    df = pd.DataFrame({
        "Date": pd.date_range("2015-01-01", periods=h.size, freq="90D"),
        "Stage Above Bed (m)": h,
        "Measured Discharge Q (m³/s)": q,
    })
    fit = fit_rating_curve(df, n_bootstrap=200, random_state=0)

    path = tmp_path / "report.xlsx"
    export_rating_curve_report(df, path, a=fit["a"], b=fit["b"], h0=fit["h0"], fit=fit)

    with zipfile.ZipFile(path) as zf:
        charts = [n for n in zf.namelist() if re.fullmatch(r"xl/charts/chart\d+\.xml", n)]
        assert charts  # the Plot + band + residual charts
        seen_any_width = False
        for name in charts:
            widths = re.findall(r'<a:ln[^>]*\bw="([^"]+)"', zf.read(name).decode())
            seen_any_width = seen_any_width or bool(widths)
            assert all(w.lstrip("-").isdigit() for w in widths), (name, widths)
        assert seen_any_width  # at least the styled line charts carry a width
        # every declared drawing part is actually present (Excel repairs strip these)
        drawings = {n for n in zf.namelist() if re.fullmatch(r"xl/drawings/drawing\d+\.xml", n)}
        assert len(drawings) == len(charts)
