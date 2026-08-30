# Rating Curve Automater

Automates building a stage–discharge **rating curve** from field measurements: it
cleans and validates a measurement workbook, fits the power-law model

```
Q = a · (H − h0)^b
```

by log–log linear regression, flags points the model fits poorly, and exports a
formatted Excel report with an embedded rating-curve chart.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### GUI

```bash
python3 rating_curve_app.py
```

Three-step wizard:

1. **Input Dataset** – browse to an `.xlsx` workbook and set the sheet name (defaults to `Measurements`), then *Upload and Validate*.
2. **Validation & Flags** – review valid / invalid / warning counts and the per-row flag list (`INVALID` rows are excluded from the fit, `WARNING` rows are kept). Optionally enter `h0` (stage of zero flow); leave blank to estimate it from the data. Choose 1 segment (single power law) or 2 (piecewise, auto breakpoint). Continue to run the regression.
3. **Export Report** – inspect the rating-curve preview (toggle log-log axes), set the uncertainty threshold, and save the multi-sheet Excel report.

### Command line

```bash
# Clean/validate any workbook to a CSV
python3 -m src.field_measurement_validation path/to/data.xlsx --sheet-name Measurements --output-csv cleaned_measurements.csv

# Clean the bundled 10-year practice dataset (either form)
python3 -m src.field_measurement_validation --default-dataset
python3 process_field_measurements.py

# Fit the curve from a cleaned CSV (single or piecewise)
python3 -m src.rating_curve_fitting
python3 -m src.rating_curve_fitting --segments 2 --h0 0.18

# Fit + write the Excel report
python3 -m src.rating_curve_report
```

## Input format

The `Measurements` sheet needs a date, a stage, and a discharge column. Column
names are matched loosely, e.g. `Date`, `Stage Above Bed (m)`,
`Measured Discharge Q (m³/s)`, plus optional `Quality` and `Field Notes`.

A row is flagged **invalid** when the date is unparseable, stage is missing or
≤ 0, discharge is missing or negative, or `Quality` reads bad/poor/unreliable/
rejected. Invalid rows are excluded from the fit and the report.

A valid row is additionally flagged with a **warning** when `Quality` reads
fair/questionable/estimated/provisional or a field note mentions backwater,
vegetation, shallow flow, ice, debris, etc. Warning rows are *kept* in the fit
but surfaced in the flag list (`has_warning` / `warning_notes` columns).

## Model notes

- `h0` (stage of zero flow) is estimated by a golden-section search that
  maximises the fit R², bounded just below the lowest observed stage. Pass an
  explicit `h0` to override.
- `segments=2` fits a piecewise curve: it searches every candidate breakpoint
  stage (each side keeping ≥ 15% of the points, min 4) for the one that
  minimises the combined residual sum of squares. `h0` is shared across
  segments. `predict_discharge(fit, stage)` evaluates either model kind.
- The fit and the report operate on the same valid-row set
  (`select_valid_measurements`), and the report reuses the fit's R².

## Project layout

| Path | Purpose |
|---|---|
| `rating_curve_app.py` | Tkinter GUI (thin view over `src/workflow.py`) |
| `process_field_measurements.py` | Batch-clean the bundled dataset |
| `src/workflow.py` | Headless load → validate → fit → export controller |
| `src/field_measurement_validation.py` | Column detection, cleaning, validation, warning tier |
| `src/rating_curve_fitting.py` | `h0` estimation and power-law fit |
| `src/rating_curve_report.py` | Excel report + chart |
| `src/rating_curve_plot.py` | Matplotlib rating-curve figure (GUI preview) |
| `tests/` | pytest suite |
| `10_year_single_site_rating_curve_data.xlsx` | Synthetic practice dataset |

## Tests

```bash
python3 -m pytest -q
```
