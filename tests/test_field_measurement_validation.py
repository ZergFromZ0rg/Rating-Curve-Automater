from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.field_measurement_validation import (
    clean_and_validate_measurements,
    clean_measurements_to_csv,
)


def test_clean_and_validate_removes_bad_rows():
    df = pd.DataFrame(
        {
            "Date": ["2020-01-01", "2020-01-02", "bad date", "2020-01-04", "2020-01-05"],
            "Stage Above Bed (m)": [0.40, -0.10, 0.55, 0.70, 0.80],
            "Measured Discharge Q (m³/s)": [0.10, 0.20, -0.30, None, 0.90],
            "Field Notes": ["Normal", "Normal", "Normal", "Normal", "Normal"],
            "Quality": ["Good", "Good", "Good", "Poor", "Good"],
        }
    )

    cleaned = clean_and_validate_measurements(df)

    assert "is_valid" in cleaned.columns
    assert cleaned["is_valid"].sum() == 2
    assert cleaned[cleaned["is_valid"]]["Date"].dt.strftime("%Y-%m-%d").tolist() == ["2020-01-01", "2020-01-05"]
    assert cleaned["validation_notes"].notna().any()


def test_warning_tier_flags_fair_quality_and_notes_without_invalidating():
    df = pd.DataFrame(
        {
            "Date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "Stage Above Bed (m)": [0.40, 0.55, 0.70],
            "Measured Discharge Q (m³/s)": [0.10, 0.20, 0.30],
            "Quality": ["Good", "Fair", "Good"],
            "Field Notes": ["Normal", "Normal", "Vegetation / possible backwater"],
        }
    )

    cleaned = clean_and_validate_measurements(df)

    assert cleaned["is_valid"].all()
    assert cleaned["has_warning"].tolist() == [False, True, True]
    assert "fair" in cleaned.loc[1, "warning_notes"].lower()
    assert "backwater" in cleaned.loc[2, "warning_notes"].lower()


def test_clean_measurements_to_csv_roundtrip(tmp_path):
    src = tmp_path / "in.xlsx"
    out = tmp_path / "out.csv"
    pd.DataFrame(
        {
            "Date": ["2020-01-01", "2020-01-02", "bad"],
            "Stage Above Bed (m)": [0.40, 0.55, 0.70],
            "Measured Discharge Q (m³/s)": [0.10, 0.20, 0.30],
        }
    ).to_excel(src, sheet_name="Measurements", index=False)

    cleaned = clean_measurements_to_csv(src, out, "Measurements")

    assert out.exists()
    on_disk = pd.read_csv(out)
    assert len(on_disk) == 3
    assert int(cleaned["is_valid"].sum()) == 2


def test_invalid_rows_do_not_carry_warnings():
    df = pd.DataFrame(
        {
            "Date": ["2020-01-01", "2020-01-02"],
            "Stage Above Bed (m)": [0.40, -0.10],
            "Measured Discharge Q (m³/s)": [0.10, 0.20],
            "Quality": ["Good", "Fair"],
            "Field Notes": ["Normal", "Shallow flow / low velocity"],
        }
    )

    cleaned = clean_and_validate_measurements(df)

    assert not cleaned.loc[1, "is_valid"]
    assert cleaned.loc[1, "warning_notes"] == ""
    assert not cleaned.loc[1, "has_warning"]
