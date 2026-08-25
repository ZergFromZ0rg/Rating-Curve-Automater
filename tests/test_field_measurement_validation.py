from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.field_measurement_validation import clean_and_validate_measurements


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
