from src.units import detect_discharge_unit, detect_stage_unit


def test_detect_stage_units():
    assert detect_stage_unit("Gauge Height (ft)").factor == 0.3048
    assert detect_stage_unit("Stage (cm)").factor == 0.01
    assert detect_stage_unit("Stage Above Bed (m)").factor == 1.0


def test_detect_discharge_units():
    cfs = detect_discharge_unit("Streamflow (cfs)")
    assert cfs.detected and abs(cfs.factor - 0.0283168) < 1e-5
    assert detect_discharge_unit("Q (m³/s)").factor == 1.0


def test_missing_unit_is_assumed_si():
    conv = detect_stage_unit("Water Level")
    assert conv.factor == 1.0
    assert conv.detected is False
    assert "assumed" in conv.label
