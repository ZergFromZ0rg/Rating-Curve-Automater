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

pip install -e .              # library + `rca` command line
pip install -e ".[app]"       # + the Streamlit web UI
pip install -e ".[app,dev]"   # + pytest   (same as: pip install -r requirements.txt)
```

Needs Python ≥ 3.10. No open-source licence is set yet: all rights reserved,
contact the author for reuse terms.

## Usage

### Web UI (recommended)

```bash
rca app          # or: streamlit run app.py
```

Opens in your browser. Upload an `.xlsx` / `.xls` / `.csv`, then work down the page:

1. **Sheet / header row** (sidebar) – blank = auto-detect.
2. **Column mapping** – each field is pre-filled from auto-detection; override any that are wrong from the dropdowns (the page re-runs live).
3. **Detected layout** – expandable panel with the chosen sheet, header row, units and a preview; opens automatically when confidence is low.
4. **Validation** – valid / invalid / warning counts, with expandable tables of the flagged rows.
5. **Fit** – site picker (if the workbook has a `Site` column), `h0` (estimate or enter), 1 or 2 segments, uncertainty threshold; shows `a` / `b` / `h0` / R² and a rating-curve plot (log-log toggle).
6. **Export** – download the multi-sheet Excel report.

The web UI and the CLI are thin views over `rating_curve_automater/workflow.py`.

### Command line

`rca <command> --help` shows each command's full options.

```bash
# Clean/validate any workbook to a CSV (sheet auto-detected unless named)
rca validate path/to/data.xlsx --output-csv cleaned_measurements.csv

# Clean the bundled 10-year practice dataset
rca validate --default-dataset

# Fit the curve from a cleaned CSV (single or piecewise; optional site filter)
rca fit
rca fit --segments 2 --h0 0.18
rca fit --site "Upper Reach"

# Fit + write the Excel report
rca report
```

## Input format

The tool accepts `.xlsx`, `.xls` and `.csv`. It needs a **date**, a **stage**
and a **discharge** column; `Quality`, `Field Notes` and `Site` are optional.
`rating_curve_automater/loader.load_measurements()` handles the variability:

- **Sheet** – if not named, each sheet is scored by how many required fields its
  header resolves and how numeric the data beneath it is; the best is chosen. If
  two sheets look equally like measurement data the report is marked
  *needs review*.
- **Header row** – title / metadata rows above the table are skipped. Each of the
  first 15 rows is scored by required-field resolution **and** whether the cells
  beneath the stage/discharge columns are actually numeric, so a section-title
  row is not mistaken for the header. Two-row headers (name row + unit row) are
  detected and combined. Pass `header_row=` (0-based) to override.
- **Column names** – matched loosely against a synonym table (`Gauge Height`,
  `Stage`, `WSE`, `GH`, `Q`, `Streamflow`, `Sample Date`, …). Ambiguities
  (e.g. two stage-like columns) are reported, not guessed silently. Extend the
  table via `config/column_aliases.yaml` (or `$RATING_CURVE_ALIASES` /
  `~/.rating_curve_automater/column_aliases.yaml`) without touching code.
- **Units** – read from the header (`(ft)`, `(cfs)`, `(cm)`, `ML/d`, …) and
  converted to SI (metres, m³/s). If no unit is found, SI is assumed and the
  report says so.
- **Wide multi-station layouts** – repeated `stage` + `discharge` column blocks
  (e.g. `SW1 readings | Stage above bed, m | Q, cms | SW2 readings | …`) are
  detected and unpivoted into a `site` column.
- **Messy values** – `"N/A"` / `"--"` placeholders become blanks; thousands
  separators and decimal commas are handled; censored values (`"<0.001"`) are
  taken at face value and flagged; `Total` / `Average` footer rows are dropped.
- **Dates** – Excel serial numbers, `DD-Mon-YY`, ambiguous day/month order,
  date ranges (`2025-10-17/22` → start date), and a separate `Time` column
  (merged into the date) are all handled.
- **Multiple sites** – a `Site` column is carried through; fit one site at a
  time (`workflow.run_fit(site=…)`, `--site`, or the GUI picker).
- **Placeholder stages** – an identical stage value repeated ≥ 3 times per site
  (a common "gauge lost / out of range" fill) is flagged invalid.
- **Overrides** – `load_measurements(path, column_overrides={"stage_m": "col_x"})`
  forces a mapping when detection is wrong.
- **Survey block** – free-text rows above the table (T-post / bed elevations,
  titles) are captured on `LoadReport.preheader_notes` for reference.

Every choice is recorded on the returned `LoadReport` (and shown in the GUI),
including `sheet_confident` / `header_confident` flags and a `needs_review`
property.

A row is flagged **invalid** when the date is unparseable, stage is missing or
≤ 0, discharge is missing or negative, `Quality` reads bad/poor/unreliable/
rejected, or its stage value repeats ≥ 3 times for the site. Invalid rows are
excluded from the fit and the report.

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
- **Plausibility check** – every fit is assessed (`assess_fit`). A **critical**
  warning (exponent `b ≤ 0`, or stage and discharge essentially uncorrelated)
  means the data is not a rating curve at all; **non-critical** warnings cover a
  low R² or fewer than 5 points. Warnings show in the web UI and the report's
  Summary sheet; `fit_rating_curve(..., strict=True)` / `--strict` raises
  `ImplausibleRatingCurve` on a critical warning.

## Project layout

| Path | Purpose |
|---|---|
| `pyproject.toml` | Package metadata, deps, the `rca` entry point |
| `app.py` | Streamlit web UI (thin view over `rating_curve_automater/workflow.py`) |
| `rating_curve_automater/cli.py` | `rca` — unified command-line entry point |
| `rating_curve_automater/workflow.py` | Headless load → validate → fit → export controller |
| `rating_curve_automater/schema.py` | Canonical column schema + header resolution (aliases from `config/column_aliases.yaml`) |
| `rating_curve_automater/units.py` | Stage/discharge unit detection and SI conversion |
| `rating_curve_automater/cleaning.py` | Messy-value coercion, footer-row drop, date/time parsing |
| `rating_curve_automater/reshape.py` | Detect + unpivot wide multi-station layouts |
| `rating_curve_automater/loader.py` | `load_measurements()` — sheet/header/column/unit detection + `LoadReport` |
| `rating_curve_automater/field_measurement_validation.py` | Validation, warning tier |
| `rating_curve_automater/rating_curve_fitting.py` | `h0` estimation and power-law fit |
| `rating_curve_automater/rating_curve_report.py` | Excel report + chart |
| `rating_curve_automater/rating_curve_plot.py` | Matplotlib rating-curve figure (GUI preview) |
| `rating_curve_automater/data/` | Bundled synthetic practice dataset (`rca validate --default-dataset`) |
| `tests/` | pytest suite |

## Tests

```bash
python3 -m pytest -q
```
