from pathlib import Path
import html
import re

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

_PKG = Path(__file__).resolve().parents[1] / "rating_curve_automater"
DATASET = _PKG / "data" / "10_year_single_site_rating_curve_data.xlsx"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _app():
    return AppTest.from_file(str(_PKG / "app.py"), default_timeout=60)


def _summary(at):
    markup = next(m.value for m in at.markdown if 'class="rca-summary-table"' in m.value)
    rows = re.findall(
        r'data-summary-label="([^"]*)" data-summary-value="([^"]*)" data-summary-note="([^"]*)"',
        markup,
    )
    return {
        html.unescape(label): html.unescape(value) + (f"\n{html.unescape(note)}" if note else "")
        for label, value, note in rows
    }


def test_app_boots_without_a_file():
    at = _app().run()
    assert not at.exception
    assert any("Turn gauging data into a rating curve" in msg.value for msg in at.markdown)


def test_app_uses_no_deprecated_streamlit_kwargs():
    src = (_PKG / "app.py").read_text()
    assert "use_container_width" not in src  # removed by Streamlit after 2025-12-31


@pytest.mark.skipif(not DATASET.exists(), reason="bundled dataset missing")
def test_app_full_run_on_bundled_dataset():
    at = _app().run()
    at.file_uploader[0].upload("data.xlsx", DATASET.read_bytes(), XLSX_MIME)
    at.run()

    assert not at.exception
    metrics = _summary(at)
    assert metrics["Valid rows"] == "120"
    assert metrics["Warnings (kept)"] == "12"
    assert 1.0 < float(metrics["a"]) < 1.4
    assert at.success and "R²" in at.success[0].value
    assert len(at.get("download_button")) == 2
    assert not at.metric
    assert "Checks" in {t.label for t in at.tabs}


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
    metrics = _summary(at)
    assert metrics["Valid rows"] == str(n)


@pytest.mark.skipif(not DATASET.exists(), reason="bundled dataset missing")
def test_app_impose_exponent_checkbox():
    at = _app().run()
    at.file_uploader[0].upload("data.xlsx", DATASET.read_bytes(), XLSX_MIME)
    at.run()

    cb = next(c for c in at.checkbox if c.label == "Impose the exponent b")
    cb.set_value(True).run()

    assert not at.exception
    value, note = _summary(at)["b"].splitlines()
    assert float(value) == pytest.approx(2.0)
    assert note == "imposed"


@pytest.mark.skipif(not DATASET.exists(), reason="bundled dataset missing")
def test_app_segmented_fit_shows_segment_summary_and_detail_tabs():
    at = _app().run()
    at.file_uploader[0].upload("data.xlsx", DATASET.read_bytes(), XLSX_MIME)
    at.run()

    shape = next(s for s in at.selectbox if s.label == "Curve shape")
    shape.set_value(2).run()

    assert not at.exception
    metrics = _summary(at)
    assert metrics["Segments"].splitlines()[0].strip() == "2"
    assert "a" not in metrics  # per-segment a/b live under Fit details instead
    assert {t.label for t in at.tabs} >= {"Rating table", "Residuals over time", "Fit details"}


@pytest.mark.parametrize("sheets, expect_concern", [(["Site A", "Site B"], True), (["Gaugings"], False)])
def test_app_columns_button_and_checks_agree(tmp_path, sheets, expect_concern):
    # Two sheets that both hold a full table -> the sheet pick is a best guess.
    n = 30
    h = np.linspace(0.3, 1.4, n)
    frame = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=n, freq="W"),
        "Stage (m)": h,
        "Discharge (m3/s)": 1.1 * (h - 0.1) ** 1.7,
    })
    path = tmp_path / "book.xlsx"
    with pd.ExcelWriter(path) as writer:
        for name in sheets:
            frame.to_excel(writer, sheet_name=name, index=False)

    at = _app().run()
    at.file_uploader[0].upload("book.xlsx", path.read_bytes(), XLSX_MIME)
    at.run()

    assert not at.exception
    labels = [p.proto.popover.label for p in at.get("popover")]
    checks = next(m.value for m in at.markdown if 'class="rca-label">Checks' in m.value)
    if expect_concern:
        assert "Columns — check" in labels
        assert "is a best guess" in checks and "Columns mapped" not in checks
    else:
        assert "Columns" in labels and "Columns — check" not in labels
        assert "Columns mapped" in checks and "best guess" not in checks
