from pathlib import Path

import pandas as pd

from src.rating_curve_report import build_observed_modeled_table, export_rating_curve_report


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
