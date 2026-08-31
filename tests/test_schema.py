import textwrap

import pandas as pd
import pytest

from rating_curve_automater.schema import (
    BUILTIN_ALIASES,
    DATE,
    DISCHARGE_CMS,
    STAGE_M,
    apply_mapping,
    ensure_canonical,
    load_alias_config,
    normalize_header,
    reload_aliases,
    resolve_columns,
)


def test_normalize_header_strips_units_and_punctuation():
    assert normalize_header("Measured Discharge Q (m³/s)") == "measured discharge q"
    assert normalize_header("Gauge_Height [ft]") == "gauge height"
    assert normalize_header(None) == ""


def test_resolve_columns_matches_varied_headers():
    mapping = resolve_columns(
        ["Sample Date", "Gauge Height (ft)", "Streamflow (cfs)", "Grade", "Remarks"]
    )
    assert mapping.is_complete
    assert mapping.fields[DATE] == "Sample Date"
    assert mapping.fields[STAGE_M] == "Gauge Height (ft)"
    assert mapping.fields[DISCHARGE_CMS] == "Streamflow (cfs)"
    assert mapping.fields["quality"] == "Grade"
    assert mapping.fields["notes"] == "Remarks"


def test_resolve_columns_reports_unresolved_required():
    mapping = resolve_columns(["Date", "Air Temperature", "Turbidity"])
    assert not mapping.is_complete
    assert STAGE_M in mapping.unresolved_required
    assert DISCHARGE_CMS in mapping.unresolved_required


def test_resolve_columns_flags_ambiguity_but_still_picks_one():
    mapping = resolve_columns(
        ["Date", "Stage Above Bed (m)", "Water Surface Elevation (masl)", "Discharge"]
    )
    assert mapping.fields[STAGE_M] == "Stage Above Bed (m)"
    assert STAGE_M in mapping.ambiguous


def test_overrides_win_and_are_not_stolen():
    mapping = resolve_columns(
        ["col_a", "col_b", "when"], overrides={STAGE_M: "col_a", DISCHARGE_CMS: "col_b", DATE: "when"}
    )
    assert mapping.fields[STAGE_M] == "col_a"
    assert mapping.fields[DISCHARGE_CMS] == "col_b"


def test_apply_mapping_renames_and_orders():
    df = pd.DataFrame({"Obs Date": ["2020-01-01"], "GH": [0.4], "Q": [0.1], "extra": [9]})
    mapping = resolve_columns(df.columns)
    out = apply_mapping(df, mapping)
    assert list(out.columns)[:3] == [DATE, STAGE_M, DISCHARGE_CMS]
    assert "extra" in out.columns


def test_alias_config_merges_user_file(tmp_path):
    cfg = tmp_path / "aliases.yaml"
    cfg.write_text(textwrap.dedent("""
        stage_m: [pool elevation, gauge board]
        discharge_cms: [adcp discharge]
        bogus_field: [ignored]
    """))

    merged = load_alias_config(cfg)
    assert "pool elevation" in merged[STAGE_M]
    assert "adcp discharge" in merged[DISCHARGE_CMS]
    assert "stage" in merged[STAGE_M]  # built-ins retained
    assert "bogus_field" not in merged

    try:
        reload_aliases(cfg)
        mapping = resolve_columns(["Date", "Pool Elevation (m)", "ADCP Discharge (m3/s)"])
        assert mapping.is_complete
        assert mapping.fields[STAGE_M] == "Pool Elevation (m)"
    finally:
        reload_aliases()


def test_alias_config_missing_file_returns_builtins():
    merged = load_alias_config("/no/such/file.yaml")
    assert merged[STAGE_M] == BUILTIN_ALIASES[STAGE_M]


def test_ensure_canonical_idempotent_and_subset_required():
    canonical = pd.DataFrame({STAGE_M: [0.4], DISCHARGE_CMS: [0.1]})
    assert ensure_canonical(canonical, required=(STAGE_M, DISCHARGE_CMS)) is canonical

    with pytest.raises(ValueError):
        ensure_canonical(pd.DataFrame({"foo": [1]}), required=(STAGE_M,))
