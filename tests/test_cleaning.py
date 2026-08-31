import numpy as np
import pandas as pd

from rating_curve_automater.cleaning import (
    clean_numeric_series,
    clean_uncertainty_series,
    coerce_datetime,
    drop_footer_rows,
    is_footer_label,
)


def test_clean_numeric_handles_placeholders_and_separators():
    s = pd.Series(["1,234.5", "N/A", "--", "  2.0 ", "1.234,56", "3,5", "", None])
    values, censored = clean_numeric_series(s)
    assert values.tolist()[:1] == [1234.5]
    assert np.isnan(values.iloc[1]) and np.isnan(values.iloc[2])
    assert values.iloc[3] == 2.0
    assert values.iloc[4] == 1234.56      # european grouped
    assert values.iloc[5] == 3.5          # decimal comma
    assert np.isnan(values.iloc[6]) and np.isnan(values.iloc[7])
    assert not censored.any()


def test_clean_numeric_flags_censored():
    values, censored = clean_numeric_series(pd.Series(["<0.001", ">5", "0.4", "~2.2"]))
    assert values.tolist() == [0.001, 5.0, 0.4, 2.2]
    assert censored.tolist() == [True, True, False, True]


def test_clean_numeric_is_idempotent_on_floats():
    values, censored = clean_numeric_series(pd.Series([0.1, 0.2, np.nan]))
    assert values.tolist()[:2] == [0.1, 0.2]
    assert not censored.any()


def test_clean_uncertainty_reads_percent_strings():
    s = pd.Series(["8%", "12 %", "±10", "N/A", "0", "-3"])
    out = clean_uncertainty_series(s)
    assert out.iloc[0] == 8.0
    assert out.iloc[1] == 12.0
    assert out.iloc[2] == 10.0             # leading ± stripped
    assert np.isnan(out.iloc[3])          # placeholder
    assert np.isnan(out.iloc[4])          # zero -> NaN (falls back to default)
    assert out.iloc[5] == 3.0             # sign of an uncertainty is meaningless


def test_clean_uncertainty_scales_fractions_to_percent():
    out = clean_uncertainty_series(pd.Series([0.05, 0.08, 0.12, 0.07]))
    assert out.tolist() == [5.0, 8.0, 12.0, 7.0]


def test_clean_uncertainty_leaves_percent_values_alone():
    out = clean_uncertainty_series(pd.Series([5.0, 8.0, 25.0]))
    assert out.tolist() == [5.0, 8.0, 25.0]


def test_footer_row_detection_and_drop():
    assert is_footer_label("Total")
    assert is_footer_label("average:")
    assert not is_footer_label("2020-01-01")

    df = pd.DataFrame({
        "date": ["2020-01-01", "2020-01-02", "Average"],
        "stage_m": [0.4, 0.5, 0.45],
        "discharge_cms": [0.1, 0.2, 0.15],
    })
    trimmed, dropped = drop_footer_rows(df, ["date"])
    assert dropped == 1
    assert len(trimmed) == 2


def test_coerce_datetime_excel_serials():
    parsed = coerce_datetime(pd.Series([43831, 43832, 43833]))  # 2020-01-01..03
    assert parsed.dt.year.tolist() == [2020, 2020, 2020]
    assert parsed.dt.day.tolist() == [1, 2, 3]


def test_coerce_datetime_dayfirst_heuristic():
    parsed = coerce_datetime(pd.Series(["25/12/2021", "13/06/2021", "01/02/2021"]))
    assert parsed.dt.month.tolist() == [12, 6, 2]


def test_coerce_datetime_dd_mon_yy_and_ranges():
    parsed = coerce_datetime(pd.Series([
        "2025-10-17/22", "15-Dec-25", "6-Mar-26", "26-May-26",
    ]))
    assert parsed.dt.year.tolist() == [2025, 2025, 2026, 2026]
    assert parsed.dt.month.tolist() == [10, 12, 3, 5]
    assert parsed.iloc[0].day == 17  # start of the range


def test_coerce_datetime_combines_separate_time_column():
    parsed = coerce_datetime(
        pd.Series(["2021-03-01", "2021-03-02"]),
        pd.Series(["09:30", "14:15:00"]),
    )
    assert parsed.iloc[0].hour == 9 and parsed.iloc[0].minute == 30
    assert parsed.iloc[1].hour == 14 and parsed.iloc[1].minute == 15
