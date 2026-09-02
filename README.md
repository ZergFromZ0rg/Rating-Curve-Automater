# Rating Curve Automater

Automates building a stage–discharge **rating curve** from field measurements: it
cleans and validates a measurement workbook, fits the power-law model

```
Q = a · (H − h0)^b
```

— by log–log least squares, or a continuous multi-segment (compound-control)
power law, or an optional Bayesian backend — with bootstrap confidence /
prediction bands, a leave-one-out accuracy figure, temporal-drift diagnostics,
an optional Manning cross-section check of the extrapolation, and a formatted
multi-sheet Excel report with charts.

> **Provisional software, no warranty.** Every rating curve, uncertainty band and
> shift flag is a decision aid — review it as a qualified hydrographer before
> operational use.

Licensed under the [BSD 3-Clause License](LICENSE).

## Install

You need a dataset of field gaugings (stage + discharge, ideally with dates) in
an `.xlsx` / `.xls` / `.csv`. Pick whichever install route matches your machine —
all of them give you the `rca` command and the `rca app` web UI. Needs Python ≥ 3.10.

**With [uv](https://docs.astral.sh/uv/) (no Python or admin rights required — `uv`
brings its own):**

```bash
uv tool install "rating-curve-automater[app]"
rca app
```

**With Anaconda / Miniconda:**

```bash
conda env create -f environment.yml
conda activate rating-curve-automater
rca app
```

**With plain pip (into a virtual environment):**

```bash
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install "rating-curve-automater[app]"
rca app
```

Add the Bayesian fit backend (heavier — `ratingcurve` + PyMC) with the `bayesian`
extra, e.g. `uv tool install "rating-curve-automater[app,bayesian]"`.

> Until the package is on PyPI, replace `rating-curve-automater` above with
> `"git+https://github.com/ZergFromZ0rg/Rating-Curve-Automater.git"` (needs `git`),
> or clone the repo and `pip install -e ".[app,dev]"` (equivalently
> `pip install -r requirements.txt`) for a contributor checkout.

## Usage

### Web UI (recommended)

```bash
rca app
```

(`rca app` just runs `streamlit run` on the packaged `app.py` for you.)

Opens in your browser. Upload an `.xlsx` / `.xls` / `.csv`, then work down the page:

1. **Sheet / header row** (sidebar) – blank = auto-detect.
2. **Column mapping** – each field is pre-filled from auto-detection; override any that are wrong from the dropdowns (the page re-runs live).
3. **Detected layout** – expandable panel with the chosen sheet, header row, units and a preview; opens automatically when confidence is low.
4. **Validation** – valid / invalid / warning counts, with expandable tables of the flagged rows.
5. **Fit** – fit method (least squares / Bayesian, plus a NUTS/ADVI sampler pick for Bayesian), site picker (if the workbook has a `Site` column), `h0` (estimate or enter), segments (1 / 2 / 3 / auto-by-BIC), discharge-uncertainty %, rating-table step, and an optional **Manning cross-section check** (upload a section + slope). Shows `a` / `b` / `h0` (with how `h0` was estimated) / R², the `b` confidence interval, and a rating-curve plot with shaded confidence / prediction bands (log-log toggle).
6. **Export** – download the multi-sheet Excel report and the stage→discharge rating table (CSV), with a preview.

If the gaugings carry dates, a **residuals-over-time** panel and a rating-shift notice appear under the plot; a Manning-check notice appears when a cross-section is given.

Both the web UI and the CLI are thin views over `rating_curve_automater/workflow.py`.

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
rca fit --segments auto            # BIC picks the segment count
rca fit --method bayesian --sampler nuts   # ratingcurve / PyMC  (needs the [bayesian] extra)
rca fit --site "Upper Reach"
rca fit --uncertainty-pct 5 --bootstrap 2000 --seed 0
rca fit --segments auto --loo      # + leave-one-out cross-validated error
rca fit --segments auto --cross-section section.csv --slope 0.0012   # + Manning check

# Fit + write the Excel report (+ optional stage->Q rating table CSV / Manning sheet)
rca report
rca report --rating-table-csv rating_table.csv --step 0.01
rca report --cross-section section.csv --slope 0.0012
```

### Library

```python
from rating_curve_automater import RatingCurveWorkflow

wf = RatingCurveWorkflow()
wf.load_and_validate("gaugings.xlsx")
outcome = wf.run_fit(segments="auto")        # fit + uncertainty bands + drift check
wf.manning_check("section.csv", slope=0.0012)  # optional: sanity-check the extrapolation
wf.export_report("report.xlsx")              # multi-sheet Excel + rating table
table = wf.rating_table(step=0.01)           # stage -> discharge DataFrame
```

## Input format

The tool accepts `.xlsx`, `.xls` and `.csv`. It needs a **date**, a **stage**
and a **discharge** column; `Quality`, `Field Notes`, `Site` and
`Discharge uncertainty` are optional.
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
- **Discharge uncertainty** – an optional per-gauging measurement-uncertainty
  column (`Discharge uncertainty (%)`, `Q uncertainty`, …) is detected and
  read as a percentage (`"8%"`, `0.08` and `8` all mean ±8 %). When present and
  its values vary, the curve is fitted by **weighted least squares** so noisier
  gaugings pull it less; a uniform column (or none) leaves the fit identical to
  plain OLS. Rows with no value fall back to the assumed default
  (`--uncertainty-pct`, `run_fit(discharge_uncertainty_pct=…)`, or the GUI
  field; default 7 %).
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

Two orthogonal choices, not a menu of overlapping ones:

* **Fit method** (`method=` / "Fit method" radio / `--method`):
  * `"ols"` (default) – log–log least squares. It is *automatically* a
    **weighted** fit when a `discharge_uncertainty` column varies, so
    "least squares" and "weighted least squares" are the same control.
  * `"bayesian"` – thodson-usgs [`ratingcurve`](https://github.com/thodson-usgs/ratingcurve)
    (a PyMC power-law model). `h0`, the segment slopes and the bands all come
    from the posterior. Because the model samples the **breakpoint stages**
    jointly with the coefficients, their uncertainty is reported too: `h0_ci` /
    `h0_sd`, and `breakpoint_ci` / `breakpoint_sd` for a segmented fit (each
    interval also on its `segments[i]` record). `assess_fit` warns when a
    breakpoint's credible interval spans more than a third of the gauged stage
    range; the plot shades each interval and the report lists them.
    `bayesian_sampler=` / `--sampler` picks `"nuts"` (exact), `"advi"` (fast
    variational) or `"auto"` (NUTS for ≤ 200 gaugings, ADVI above); NUTS places
    breakpoints more reliably. Needs the `bayesian` extra (see [Install](#install)).
* **Shape** (`segments=`) – `1`, an integer `≥ 2`, or `"auto"`. This is the
  "segmented power law with breakpoints" option and it composes with either
  method. `"auto"` fits 1..N segments and keeps the lowest-BIC count — for the
  Bayesian backend too (each candidate is a full fit, so it is slow).

- `h0` (stage of zero flow) is estimated by the classic three-point
  (geometric-progression) method, refined by the `h0` that drives the log–log
  **curvature** of the low-flow gaugings to zero, with a reliability gate that
  falls back to a neutral default when the three-point triples disagree. Unlike
  maximising R² (which collapses toward `h0 → 0`), this does not trade `h0` off
  against `a` and `b`. `fit["h0_diagnostics"]` records the `method`, whether the
  estimate `railed` against a bound (→ a "weakly identified" warning) and the
  residual curvature. Pass an explicit `h0` to override; the Bayesian backend
  infers `h0` from the posterior instead.
- **Weighting** – the log–log regression weights each gauging by
  `1 / (fractional discharge uncertainty)`. With no uncertainty column and one
  assumed percentage the weights are uniform (ordinary least squares); a
  varying per-point column makes it a genuine weighted fit. `h0` estimation and
  the piecewise fit use the same weights. The fit dict records
  `uncertainty_source`, `mean_uncertainty_pct` and `weighted`. A weighted fit is
  also judged on a **weighted R²** (`r_squared_weighted`), so a deliberately
  down-weighted outlier no longer trips the "poor fit" warning.
- **Confidence & prediction bands** – `fit_rating_curve(..., n_bootstrap=1000)`
  (on by default in the web UI / `--bootstrap`) runs a wild residual bootstrap in
  log space, re-fitting the curve with the same weights. When `h0` was estimated
  it is **re-estimated inside every replicate** (physical breakpoints held), so
  the band carries the point-of-zero-flow uncertainty — otherwise the low-flow
  band is too tight. It returns `fit["bands"]`: a dense stage grid with a
  **confidence** band (how well the mean curve is known) and a wider
  **prediction** band (where the next gauging would fall), plus `ci_level`
  intervals on `a`, `b` and `h0`. The band spans the observed stage range only –
  it is not an extrapolation tool. The web UI shades it on the plot and the Excel
  report gets a **Rating Curve Band** sheet. Needs ≥ 4 usable gaugings.
- **Leave-one-out accuracy** – `leave_one_out_error(df, …)` (in the Excel report,
  and `rca fit --loo`) holds out each valid gauging, re-fits, and predicts it;
  reports out-of-sample RMSPE / bias / 95th-percentile error — an honest figure
  where the in-sample R² is not. Needs ≥ 8 valid gaugings.
- **Temporal drift / rating shift** – when the gaugings carry dates,
  `assess_temporal_drift` (always run; `fit["drift"]`) fits the curve's
  log-residuals against gauging date and runs three permutation tests: a linear
  **time trend** (`trend_pct_per_year`); whether the most recent ~12 months of
  gaugings sit systematically off the curve (`recent_mean_pct`); and a
  **split-period test** — an early-half vs late-half re-fit compared over their
  shared stage range (`split_shift_pct`) — which catches a shift the combined fit
  absorbed into curve shape. The `flag` is `none` / `possible` / `likely`, or **`unassessable`** when
  stage and gauging date are so rank-correlated (`stage_time_corr`) that a shift
  cannot be separated from the curve shape. A material trend (≥ 3 %/yr), recent
  bias (≥ 7 %) or split shift (≥ 7 %) that is also significant → `likely`, with a
  message pointing at scour vs aggradation and suggesting a stage shift
  or re-fit. When the flag is `possible` or `likely` it also locates the single
  most likely **changepoint** (`fit["drift"]["changepoint"]`): the split date
  whose two separate rating curves best fit the record, with the percent shift
  across the break (`shift_pct`), the gauging counts on each side, and a
  permutation p-value that accounts for having scanned every candidate date. The
  web UI shows a residual-vs-date plot and the changepoint estimate; the report
  gets changepoint rows in the summary plus a **Residuals Over Time** sheet (with
  the per-gauging percent difference and stage shift). Needs ≥ 6 dated gaugings
  over ≥ 45 days.
- **Rating table** – `build_rating_table(fit, step=0.01, stage_min=…, stage_max=…)`
  tabulates `Q` (and the confidence / prediction bounds) against stage on a
  fixed grid – the lookup table applied to a continuous stage record. It
  defaults to the gauged stage range; rows outside it are kept but flagged
  `Within gauged range = False` and their band columns left blank. Ships as the
  **Rating Table** sheet in the Excel report and a standalone CSV
  (`export_rating_table_csv`, `--rating-table-csv`, or the GUI download).
- **Piecewise (multi-control) curve** – `segments=N` (N ≥ 2) fits a
  **continuous** piecewise power law: a linear spline in
  `(ln(H − h0), ln Q)` space with N − 1 breakpoints chosen by forward
  selection (each segment keeps ≥ max(4, 10%) of the gaugings across ≥ 2
  distinct stages). `segments="auto"` tries 1..`max_segments` (default 4) and
  keeps the count that minimises BIC (or `segment_criterion="aic"`), so the
  curve only gains a segment it can justify — on the bundled single-control
  dataset auto returns 1 segment. The curve has no jump at a breakpoint. The
  fit carries `breakpoints` (list), `n_segments`, and a `segments` list of
  per-segment `a`/`b`; `predict_discharge(fit, stage)` evaluates any kind
  (`rating_curve_automater/piecewise.py`).
- **Manning cross-section check** (optional) – give a surveyed cross-section
  (offset + elevation CSV, or arrays) and the channel slope and
  `manning_sanity_check(fit, offset, bed, slope, n=None)` /
  `RatingCurveWorkflow.manning_check(...)` / `rca fit --cross-section xs.csv
  --slope 0.001` (also `--mannings-n`, `--stage-offset`) computes an independent
  **Manning** curve `Q = (1/n)·A·R^(2/3)·S^(1/2)`. Manning's `n` is calibrated
  to the rating over the gauged range (so the check is about the *shape* of the
  extrapolation, not a guessed roughness) unless you pass one. The result
  carries both curves on a grid, their percent difference, and a `flag`:
  `ok` / `check` (15–35 % drift) / `diverges` (> 35 % — the power-law
  extrapolation above the highest gauging is not supported by the channel
  geometry) / `implausible-n` (the calibrated `n` is outside 0.015–0.20 → the
  slope, survey or rating is off) / `unusable` (bad inputs). Shows on the rating
  plot, the web UI, and a **Manning Check** sheet in the report. Single `n`,
  steady uniform flow — most trustworthy for in-bank / slightly-out-of-bank
  extrapolation.
- The fit and the report operate on the same valid-row set
  (`select_valid_measurements`), and the report reuses the fit's R².
- **Plausibility check** – every fit is assessed (`assess_fit`). A **critical**
  warning (exponent `b ≤ 0`, or stage and discharge essentially uncorrelated)
  means the data is not a rating curve at all; **non-critical** warnings cover a
  low R², fewer than 5 points, a weakly-identified `h0` (the estimate hit a
  search bound), and a single power law fitted to gaugings that are visibly
  curved in log–log space (→ "try `segments=2` / `"auto"`"). Warnings show in the
  web UI and the report's Summary sheet; `fit_rating_curve(..., strict=True)` /
  `--strict` raises `ImplausibleRatingCurve` on a critical warning.

## Project layout

| Path | Purpose |
|---|---|
| `pyproject.toml` | Package metadata, deps, the `rca` entry point |
| `environment.yml` | Conda one-file setup |
| `rating_curve_automater/app.py` | Streamlit web UI (thin view over `workflow.py`; launched by `rca app`) |
| `rating_curve_automater/cli.py` | `rca` — unified command-line entry point |
| `rating_curve_automater/workflow.py` | Headless load → validate → fit → export controller |
| `rating_curve_automater/schema.py` | Canonical column schema + header resolution (aliases from `config/column_aliases.yaml`) |
| `rating_curve_automater/units.py` | Stage/discharge unit detection and SI conversion |
| `rating_curve_automater/cleaning.py` | Messy-value coercion, footer-row drop, date/time parsing |
| `rating_curve_automater/reshape.py` | Detect + unpivot wide multi-station layouts |
| `rating_curve_automater/loader.py` | `load_measurements()` — sheet/header/column/unit detection + `LoadReport` |
| `rating_curve_automater/field_measurement_validation.py` | Validation, warning tier |
| `rating_curve_automater/rating_curve_fitting.py` | `h0` estimation, (weighted) power-law fit, plausibility check |
| `rating_curve_automater/rating_curve_uncertainty.py` | Bootstrap confidence / prediction bands |
| `rating_curve_automater/piecewise.py` | Continuous N-segment power law, BIC segment selection |
| `rating_curve_automater/bayesian.py` | Optional Bayesian backend (wraps `ratingcurve` / PyMC) |
| `rating_curve_automater/rating_curve_drift.py` | Residual-vs-date trend, split-period test, changepoint date/magnitude + rating-shift flag |
| `rating_curve_automater/manning.py` | Manning cross-section sanity check of the curve's extrapolation |
| `rating_curve_automater/rating_table.py` | Stage→discharge lookup table (Excel sheet + CSV) |
| `rating_curve_automater/rating_curve_report.py` | Excel report + chart |
| `rating_curve_automater/rating_curve_plot.py` | Matplotlib rating-curve figure (GUI preview) |
| `rating_curve_automater/data/` | Bundled synthetic practice dataset (`rca validate --default-dataset`) |
| `tests/` | pytest suite |

## Tests

```bash
python3 -m pytest -q
```
