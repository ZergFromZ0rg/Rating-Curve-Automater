from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

_PKG = Path(__file__).resolve().parents[1] / "rating_curve_automater"
DATASET = _PKG / "data" / "10_year_single_site_rating_curve_data.xlsx"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _app():
    return AppTest.from_file(str(_PKG / "app.py"), default_timeout=60)


def test_app_boots_without_a_file():
    at = _app().run()
    assert not at.exception
    assert any("Upload" in msg.value for msg in at.info)


def test_app_uses_no_deprecated_streamlit_kwargs():
    src = (_PKG / "app.py").read_text()
    assert "use_container_width" not in src  # removed by Streamlit after 2025-12-31


@pytest.mark.skipif(not DATASET.exists(), reason="bundled dataset missing")
def test_app_full_run_on_bundled_dataset():
    at = _app().run()
    at.file_uploader[0].upload("data.xlsx", DATASET.read_bytes(), XLSX_MIME)
    at.run()

    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Valid rows"] == "120"
    assert metrics["Warnings (kept)"] == "12"
    assert 1.0 < float(metrics["a"]) < 1.4
    assert at.success and "R²" in at.success[0].value
    assert at.get("download_button")


@pytest.mark.skipif(not DATASET.exists(), reason="bundled dataset missing")
def test_app_column_override(tmp_path):
    path = tmp_path / "cryptic.xlsx"
    n = 30
    h = np.linspace(0.3, 1.4, n)
    pd.DataFrame({
        "d": pd.date_range("2020-01-01", periods=n, freq="W"),
        "x1": h,
        "x2": 1.1 * (h - 0.1) ** 1.7,
    }).to_excel(path, sheet_name="S", index=False)

    at = _app().run()
    at.file_uploader[0].upload("cryptic.xlsx", path.read_bytes(), XLSX_MIME)
    at.run()

    # auto-detection can't resolve x1/x2 -> an error is shown
    assert at.error

    by_key = {sb.key: sb for sb in at.selectbox}
    by_key["map_date"].set_value("d").run()
    by_key["map_stage_m"].set_value("x1").run()
    by_key["map_discharge_cms"].set_value("x2").run()

    assert not at.error
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Valid rows"] == str(n)
