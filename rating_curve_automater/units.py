"""Detect stage/discharge units from column headers and convert to SI.

Conversions are applied by :func:`rating_curve_automater.loader.load_measurements`. When no unit
can be read from the header the value is assumed to already be in SI
(metres / cubic metres per second) and the report records that assumption.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

STAGE_SI = "m"
DISCHARGE_SI = "m³/s"

# normalised token -> (display label, factor to SI)
_STAGE_UNITS: dict[str, tuple[str, float]] = {
    "m": ("m", 1.0),
    "meter": ("m", 1.0),
    "meters": ("m", 1.0),
    "metre": ("m", 1.0),
    "metres": ("m", 1.0),
    "masl": ("m asl", 1.0),
    "mahd": ("m AHD", 1.0),
    "cm": ("cm", 0.01),
    "mm": ("mm", 0.001),
    "ft": ("ft", 0.3048),
    "feet": ("ft", 0.3048),
    "foot": ("ft", 0.3048),
    "in": ("in", 0.0254),
    "inch": ("in", 0.0254),
    "inches": ("in", 0.0254),
}

_DISCHARGE_UNITS: dict[str, tuple[str, float]] = {
    "m3 s": ("m³/s", 1.0),
    "m3s": ("m³/s", 1.0),
    "cms": ("m³/s", 1.0),
    "cumec": ("m³/s", 1.0),
    "cumecs": ("m³/s", 1.0),
    "ft3 s": ("ft³/s", 0.028316846592),
    "ft3s": ("ft³/s", 0.028316846592),
    "cfs": ("ft³/s", 0.028316846592),
    "cusec": ("ft³/s", 0.028316846592),
    "cusecs": ("ft³/s", 0.028316846592),
    "l s": ("L/s", 0.001),
    "ls": ("L/s", 0.001),
    "lps": ("L/s", 0.001),
    "l min": ("L/min", 1.0 / 60000.0),
    "ml d": ("ML/d", 1.0e6 / 86400.0 / 1000.0),
    "mld": ("ML/d", 1.0e6 / 86400.0 / 1000.0),
    "m3 d": ("m³/d", 1.0 / 86400.0),
    "m3 h": ("m³/h", 1.0 / 3600.0),
    "gpm": ("US gpm", 6.30902e-5),
    "mgd": ("MGD", 0.0438126364),
}

_BRACKET_RE = re.compile(r"[\(\[\{]([^)\]\}]*)[\)\]\}]")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class UnitConversion:
    label: str          # unit read from the header, or the assumed SI unit
    factor: float       # multiply raw values by this to get SI
    detected: bool      # False => SI was assumed


def _normalise_unit_text(text: str) -> str:
    text = text.lower().replace("³", "3").replace("²", "2").replace("/", " ")
    text = _NON_ALNUM_RE.sub(" ", text)
    return " ".join(text.split())


def _lookup(header: object, table: dict[str, tuple[str, float]], si_label: str) -> UnitConversion:
    if header is not None:
        match = _BRACKET_RE.search(str(header))
        candidates = []
        if match:
            candidates.append(_normalise_unit_text(match.group(1)))
        # also try trailing token(s) outside brackets, e.g. "Gauge height ft"
        tail = _normalise_unit_text(re.sub(_BRACKET_RE, " ", str(header)))
        if tail:
            candidates.append(tail.split()[-1])
        for cand in candidates:
            if cand in table:
                label, factor = table[cand]
                return UnitConversion(label=label, factor=factor, detected=True)
    return UnitConversion(label=f"assumed {si_label}", factor=1.0, detected=False)


def detect_stage_unit(header: object) -> UnitConversion:
    return _lookup(header, _STAGE_UNITS, STAGE_SI)


def detect_discharge_unit(header: object) -> UnitConversion:
    return _lookup(header, _DISCHARGE_UNITS, DISCHARGE_SI)
