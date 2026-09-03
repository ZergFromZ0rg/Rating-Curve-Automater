from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rating_curve_automater.field_measurement_validation import (
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
    assert cleaned[cleaned["is_valid"]]["date"].dt.strftime("%Y-%m-%d").tolist() == ["2020-01-01", "2020-01-05"]
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


def test_clean_measurements_to_csv_honours_column_overrides(tmp_path):
    src = tmp_path / "in.xlsx"
    out = tmp_path / "out.csv"
    pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=4),
        "Staff gauge (m)": [0.30, 0.40, 0.50, 0.60],
        "Surveyed level (m)": [12.30, 12.40, 12.55, 12.70],
        "Q (m3/s)": [0.05, 0.10, 0.20, 0.30],
    }).to_excel(src, sheet_name="Measurements", index=False)

    cleaned = clean_measurements_to_csv(
        src, out, "Measurements", column_overrides={"stage_m": "Surveyed level (m)"}
    )

    assert cleaned["stage_m"].tolist() == [12.30, 12.40, 12.55, 12.70]


def test_repeated_stage_value_is_flagged_as_placeholder():
    df = pd.DataFrame({
        "Date": pd.date_range("2025-01-01", periods=6, freq="MS"),
        "Stage Above Bed (m)": [0.5, 0.7, 0.9, 1.212, 1.212, 1.212],
        "Measured Discharge Q (m³/s)": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    })

    cleaned = clean_and_validate_measurements(df)

    assert cleaned["is_valid"].tolist() == [True, True, True, False, False, False]
    assert "repeats 3x" in cleaned.loc[3, "validation_notes"]


def test_recurring_stage_with_varying_discharge_is_kept():
    # A common low-flow stage revisited years apart, each visit a genuine (and
    # different) discharge — not a stuck gauge. Must stay valid.
    df = pd.DataFrame({
        "Date": ["2008-02-19", "2008-06-01", "2009-03-10",
                 "2011-11-19", "2014-07-01", "2016-01-14"],
        "Stage Above Bed (m)": [0.26, 0.31, 0.19, 0.26, 0.44, 0.26],
        "Measured Discharge Q (m³/s)": [0.090, 0.11, 0.05, 0.044, 0.02, 0.128],
    })

    cleaned = clean_and_validate_measurements(df)

    assert cleaned["is_valid"].all()


def test_repeated_stage_and_discharge_is_flagged_as_duplicate():
    df = pd.DataFrame({
        "Date": ["2020-01-01", "2021-06-01", "2023-09-01", "2019-01-01"],
        "Stage Above Bed (m)": [0.50, 0.50, 0.50, 0.72],
        "Measured Discharge Q (m³/s)": [0.20, 0.20, 0.20, 0.55],
    })

    cleaned = clean_and_validate_measurements(df)

    assert cleaned["is_valid"].tolist() == [False, False, False, True]
    assert "discharge both repeat" in cleaned.loc[0, "validation_notes"]


def test_repeated_stage_is_per_site():
    df = pd.DataFrame({
        "Date": list(pd.date_range("2025-01-01", periods=4, freq="MS")) * 2,
        "Site": ["A"] * 4 + ["B"] * 4,
        "Stage Above Bed (m)": [1.0, 1.0, 1.0, 0.9, 0.4, 0.5, 0.6, 0.7],
        "Measured Discharge Q (m³/s)": [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4],
    })

    cleaned = clean_and_validate_measurements(df)
    # site A's three identical 1.0 rows are dropped; site B untouched
    assert cleaned.loc[cleaned["site"] == "A", "is_valid"].tolist() == [False, False, False, True]
    assert cleaned.loc[cleaned["site"] == "B", "is_valid"].all()


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
