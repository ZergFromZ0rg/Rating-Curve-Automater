"""Rating Curve Automater — build a stage–discharge rating curve from messy
field-gauging spreadsheets.

Typical use::

    from rating_curve_automater import RatingCurveWorkflow

    wf = RatingCurveWorkflow()
    wf.load_and_validate("gaugings.xlsx")
    outcome = wf.run_fit()
    wf.export_report("report.xlsx")

or piece by piece with :func:`load_measurements`,
:func:`clean_and_validate_measurements`, :func:`fit_rating_curve` and
:func:`predict_discharge`.
"""

from __future__ import annotations

__version__ = "0.1.0"

#: Provisional software. Fits and diagnostics are decision aids, not a
#: substitute for review by a qualified hydrographer; no warranty is expressed
#: or implied, and results may change between versions.
DISCLAIMER = (
    "Rating Curve Automater is provisional software provided without warranty. "
    "Every rating curve it produces should be reviewed by a qualified "
    "hydrographer before operational use."
)

from rating_curve_automater.field_measurement_validation import (  # noqa: E402
    clean_and_validate_measurements,
)
from rating_curve_automater.loader import LoadReport, load_measurements  # noqa: E402
from rating_curve_automater.rating_curve_fitting import (  # noqa: E402
    fit_rating_curve,
    predict_discharge,
)
from rating_curve_automater.workflow import RatingCurveWorkflow  # noqa: E402

__all__ = [
    "__version__",
    "DISCLAIMER",
    "RatingCurveWorkflow",
    "load_measurements",
    "LoadReport",
    "clean_and_validate_measurements",
    "fit_rating_curve",
    "predict_discharge",
]
