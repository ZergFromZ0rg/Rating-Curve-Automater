<div align="center">

# Rating Curve Automater

**Turn field-gauging spreadsheets into validated, uncertainty-aware
stage–discharge rating curves.**

[Web UI](#getting-started) · [CLI](#commands) · [Python API](#python-api) · [Releases](https://github.com/ZergFromZ0rg/Rating-Curve-Automater/releases)

[![PyPI](https://img.shields.io/pypi/v/rating-curve-automater)](https://pypi.org/project/rating-curve-automater/)
[![Python](https://img.shields.io/pypi/pyversions/rating-curve-automater)](https://pypi.org/project/rating-curve-automater/)
[![Tests](https://github.com/ZergFromZ0rg/Rating-Curve-Automater/actions/workflows/test.yml/badge.svg)](https://github.com/ZergFromZ0rg/Rating-Curve-Automater/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)](LICENSE)

</div>

Rating Curve Automater cleans and validates field measurements, fits rating
curves, reports uncertainty and drift diagnostics, and exports an Excel report.

> Provisional software: review curves, flags, and extrapolations as a qualified
> hydrographer before operational use.

![Rating Curve Automater interface preview](docs/images/rating-curve-automater-ui-preview.png)

## Getting started

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install "rating-curve-automater[app]"
rca app
```

Upload an `.xlsx`, `.xls`, or `.csv` file containing date, stage, and discharge
measurements. The app validates the data, fits the curve, shows diagnostics,
and provides Excel and rating-table downloads.

For a development checkout:

```bash
git clone https://github.com/ZergFromZ0rg/Rating-Curve-Automater.git
cd Rating-Curve-Automater
pip install -e ".[app,dev]"
```

## What it does

- Handles messy spreadsheet layouts, units, missing values, quality flags, and
  multiple sites.
- Supports OLS, weighted, piecewise, and optional Bayesian fits.
- Produces confidence/prediction bands, leave-one-out checks, drift diagnostics,
  charts, and Excel reports.

## Commands

```bash
# Validate measurements
rca validate measurements.xlsx --output-csv cleaned.csv

# Fit a curve
rca fit --csv cleaned.csv --segments auto --loo

# Export an Excel report and rating table
rca report --csv cleaned.csv --output rating_curve_report.xlsx \
  --rating-table-csv rating_table.csv
```

Run `rca <command> --help` for all options. Install the optional Bayesian
backend with `pip install "rating-curve-automater[bayesian]"`.

## Python API

```python
from rating_curve_automater import RatingCurveWorkflow

workflow = RatingCurveWorkflow()
workflow.load_and_validate("measurements.xlsx")
workflow.run_fit(segments="auto")
workflow.export_report("rating_curve_report.xlsx")
```

## Development

```bash
python -m pytest -q
python -m compileall -q rating_curve_automater
python -m build --no-isolation
```

CI runs tests on Python 3.10–3.13, checks lint, and builds the package on pull
requests and pushes to `main`.

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md),
[PUBLISHING.md](PUBLISHING.md), and [CHANGELOG.md](CHANGELOG.md).

## License

BSD 3-Clause. See [LICENSE](LICENSE).
