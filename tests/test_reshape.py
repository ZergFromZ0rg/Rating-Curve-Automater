import numpy as np
import pandas as pd

from rating_curve_automater.schema import DATE, DISCHARGE_CMS, SITE, STAGE_M
from rating_curve_automater.loader import load_measurements
from rating_curve_automater.reshape import detect_wide_blocks, reshape_wide_stations


def _wide_frame():
    return pd.DataFrame({
        "Date": pd.date_range("2025-01-01", periods=4, freq="MS"),
        "SW1 readings": [0.34, 0.59, 0.63, 0.55],
        "Stage above bed, m": [0.87, 0.62, 0.58, 0.66],
        "Q, cms": [0.05, 0.13, 0.52, 0.30],
        "SW2 readings": [0.31, 0.70, 0.48, 0.44],
        "Stage above bed, m.1": [0.82, 0.43, 0.65, 0.70],
        "Q, cms.1": [0.05, 0.11, 0.53, 0.31],
    })


def test_detect_wide_blocks_names_stations():
    blocks = detect_wide_blocks(list(_wide_frame().columns))
    assert blocks is not None
    names = [b[0] for b in blocks]
    assert names == ["SW1", "SW2"]


def test_reshape_wide_stations_to_long():
    long_df, names, unit_ref = reshape_wide_stations(_wide_frame())
    assert names == ["SW1", "SW2"]
    assert set(long_df.columns) == {DATE, STAGE_M, DISCHARGE_CMS, SITE}
    assert len(long_df) == 8
    assert long_df.loc[long_df[SITE] == "SW2", STAGE_M].tolist() == [0.82, 0.43, 0.65, 0.70]
    assert unit_ref[DISCHARGE_CMS] == "Q, cms"


def test_single_station_is_not_reshaped():
    df = pd.DataFrame({
        "Date": pd.date_range("2025-01-01", periods=3),
        "Stage (m)": [0.4, 0.5, 0.6],
        "Q (m3/s)": [0.1, 0.2, 0.3],
    })
    assert reshape_wide_stations(df) is None


def test_load_measurements_unpivots_wide_workbook(tmp_path):
    path = tmp_path / "wide.xlsx"
    rows = [
        [None, None, 2026, None, None, None, None],
        ["SW1", "Top of T-Post", 336.99, None, None, None, None],
        [None, "Bed elevation", 335.778, None, None, None, None],
        [None, None, None, None, None, None, None],
        ["Date", "SW1 readings", "Stage above bed, m", "Q, cms", "SW2 readings", "Stage above bed, m", "Q, cms"],
    ]
    h = np.linspace(0.3, 1.2, 8)
    for i in range(8):
        d = f"2025-{i+1:02d}-05"
        rows.append([d, 0.5, h[i], 1.1 * (h[i] - 0.1) ** 1.7, 0.5, h[i] + 0.05, 1.0 * (h[i]) ** 1.6])
    pd.DataFrame(rows).to_excel(path, sheet_name="Sheet1", index=False, header=False)

    df, report = load_measurements(path)
    assert report.ok
    assert SITE in df.columns
    assert sorted(df[SITE].unique()) == ["SW1", "SW2"]
    assert any("wide multi-station" in m for m in report.messages)
    assert any("Top of T-Post" in n for n in report.preheader_notes)
