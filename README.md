# Rating Curve Automater

[![PyPI](https://img.shields.io/pypi/v/rating-curve-automater)](https://pypi.org/project/rating-curve-automater/)
[![Python](https://img.shields.io/pypi/pyversions/rating-curve-automater)](https://pypi.org/project/rating-curve-automater/)
[![Tests](https://github.com/ZergFromZ0rg/Rating-Curve-Automater/actions/workflows/test.yml/badge.svg)](https://github.com/ZergFromZ0rg/Rating-Curve-Automater/actions/workflows/test.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue)](LICENSE)

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
conda create -n rca "python>=3.10" pip
conda activate rca
pip install "rating-curve-automater[app]"
rca app
```

(Or, from a repo clone, `conda env create -f environment.yml`.)

**With plain pip (into a virtual environment):**

```bash
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install "rating-curve-automater[app]"
rca app
```

Add the Bayesian fit backend (heavier — `ratingcurve` + PyMC) with the `bayesian`
extra, e.g. `uv tool install "rating-curve-automater[app,bayesian]"`.

> **From source** (contributors): clone the repo and `pip install -e ".[app,dev]"`
> (equivalently `pip install -r requirements.txt`).

## Usage

### Web UI (recommended)

```bash
rca app
```

(`rca app` just runs `streamlit run` on the packaged `app.py` for you.)

Opens in your browser as **one column, top to bottom — no sidebar.**

1. **Upload** the workbook.
2. **Detected layout** – a collapsed panel that opens itself (`⚠️`) only when
   auto-detection is unsure: sheet / header row, and the column mapping (three
   starred required fields in a row; optional ones behind a checkbox).
3. **Fit** – the choices that change the curve are all on screen: curve shape
   (single / 2 / 3 / auto-by-BIC), method (least squares / Bayesian + sampler),
   and two toggles — *set `h₀` by hand* and *impose the exponent `b`*. Two
   drawers hold the rest: *Uncertainty & point flags* (assumed discharge
   uncertainty %, the "mark a gauging uncertain" threshold) and *Advanced*
   (rating-table step, the Manning cross-section check).
4. **Result** – a valid / excluded / flagged row summary with drill-downs; a
   plausibility banner and the fitted equation; `a` / `b` / `h₀` / R² with one
   line of context (gaugings used, how `h₀` came about, the `b` CI, band width);
   the rating-curve plot with shaded confidence / prediction bands (log-log
   toggle); a *Diagnostics* section (temporal-drift verdict, changepoint,
   residuals-over-time plot, Manning check); and the Excel-report / rating-table
   downloads with a preview.

Both the web UI and the CLI are thin views over `rating_curve_automater/workflow.py`.

### Command line

`rca <command> --help` shows each command's full options.

```bash
# Clean/validate any workbook to a CSV (sheet auto-detected unless named)
rca validate path/to/data.xlsx --output-csv cleaned_measurements.csv

# Force a column when two look alike (e.g. a staff gauge vs a surveyed level)
rca validate data.xlsx --stage-column "Elev (masl)"

# Clean the bundled 10-year practice dataset
rca validate --default-dataset

# Fit the curve from a cleaned CSV (single or piecewise; optional site filter)
rca fit
rca fit --segments 2 --h0 0.18
rca fit --segments auto            # BIC picks the segment count
rca fit --exponent 2.0 --h0 0.18   # impose b (section control); fit only a
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
  (e.g. two stage-like columns) are reported, not guessed silently, and the
  message names the override to use (`rca validate --stage-column …` /
  `--discharge-column …`, `column_overrides=`, or the GUI dropdown). Extend the
  synonym table via `config/column_aliases.yaml` (or `$RATING_CURVE_ALIASES` /
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
- **Placeholder stages / stuck gauge** – a stage value repeated ≥ 3 times per
  site is flagged invalid **only** when it also looks non-physical: the
  discharge repeats too (a duplicated / placeholder row) or the readings are an
  unbroken run of consecutive visits (the gauge stuck at one value). A stage
  that merely recurs across the record with genuinely different discharges (a
  common low-flow stage) is kept.
- **Discharge uncertainty** – an optional per-gauging measurement-uncertainty
  column (`Discharge uncertainty (%)`, `Q uncertainty`, …) is detected and
  read as a percentage (`"8%"`, `0.08` and `8` all mean ±8 %). When present and
  its values vary, the curve is fitted by **weighted least squares** so noisier
  gaugings pull it less; a uniform column (or none) leaves the fit identical to
  plain OLS. Rows with no value fall back to the assumed default
  (`--uncertainty-pct`, `run_fit(discharge_uncertainty_pct=…)`, or the GUI
  field; default 7 %).
- **Overrides** – `load_measurements(path, column_overrides={"stage_m": "col_x"})`,
  `rca validate --stage-column / --discharge-column / --date-column`, or the GUI
  dropdown force a mapping when detection is wrong.
- **Survey block** – free-text rows above the table (T-post / bed elevations,
  titles) are captured on `LoadReport.preheader_notes` for reference.

Every choice is recorded on the returned `LoadReport` (and shown in the GUI),
including `sheet_confident` / `header_confident` flags and a `needs_review`
property.

A row is flagged **invalid** when the date is unparseable, stage is missing or
≤ 0, discharge is missing or negative, `Quality` reads bad/poor/unreliable/
rejected, or its stage value repeats in a way that looks like a placeholder or a
stuck gauge (see *Placeholder stages / stuck gauge* above). Invalid rows are
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
  infers `h0` from the posterior instead. With `segments="auto"` and an
  estimated `h0`, the single-power-law baseline the breakpoints have to beat is
  re-profiled for `h0` within ±0.10 m (`piecewise.H0_REFINE_WINDOW_M`): a knot
  adds the same freedom, so without this a small `h0` error bends the log–log
  line and the search "fixes" the bend with a spurious segment. A one-power-law
  channel then keeps 1 segment (and reports the refined `h0`); a genuine
  compound channel still splits.
- **Imposed exponent** – `fit_rating_curve(..., fixed_b=2.0)` / `rca fit
  --exponent 2.0` / the "Impose the exponent b" checkbox pins `b` to a value from the
  control type (≈ 1.5 broad-crested weir, ≈ 2–2.5 natural section control, ≈ 2.5
  V-notch) and fits **only** `a` (and `h0`, if not supplied). Use it when a
  narrow-range or scattered low-flow record cannot identify the exponent on its
  own — a free fit then returns a meaningless `b` and is rejected. Single power
  law, `method="ols"` only; the fit carries `b_fixed`, the bootstrap reports a
  CI on `a` but not `b`, and an uncorrelated stage/discharge cloud becomes a
  "treat as provisional" warning rather than a hard "not a rating curve".
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
| `.github/workflows/` | CI: `test.yml` (pytest on 3.10–3.13 + a pyflakes lint, every push/PR), `publish.yml` (PyPI on GitHub Release) |
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
| `benchmarks/` | Seeded Monte-Carlo validation study (`simulation_study.py`) + its results |

## Tests

```bash
python3 -m pytest -q
```

CI runs this on Python 3.10–3.13 (plus a pyflakes lint) on every push and pull
request; the optional `[bayesian]` tests self-skip unless PyMC is installed.

## Validation

Beyond the unit tests, the estimator is checked against a seeded Monte-Carlo
study — `benchmarks/simulation_study.py` (`--quick` for a ~1-minute run,
`benchmarks/README.md` for the design). It builds stage–discharge curves with
**known** parameters, adds unbiased multiplicative noise, fits them through the
same `fit_rating_curve` path a user runs, and scores parameter recovery and
interval calibration. Latest full run
(`benchmarks/results/simulation_study.json`):

- **Single power law** (n = 25, ~8 % noise, 400 trials) — exponent `b`:
  RMSE ≈ 10 %, bias ≈ −2 %; `h0`: RMSE ≈ 0.043 m, bias ≈ 0.6 cm. The 95 %
  bootstrap CIs on `a`, `b`, `h0` cover the truth ~99 % of the time (slightly
  conservative); the 95 % confidence band has 96 % pointwise coverage at a
  median half-width of ±11 %.
- **Segment count** (`segments="auto"`, BIC; ~7 % noise, n = 60–75) — one true
  control: keeps 1 segment **98.8 %** of trials whether `h0` is supplied or
  estimated (up from ~52 % with an estimated `h0` before the v0.3.1 refinement).
  Two true controls: exact count 88 % (estimated `h0`) / 96 % (supplied), with
  breakpoints located to **~1 % of the gauged stage range**. Three controls is
  hard — at least 2 segments every time, exact count ~23 %.
- **Temporal drift / rating shift** (dated gaugings, 5-year span) — **5 %**
  false-positive rate at no trend; gradual drift detected in **88 %** of trials
  at ≥ 4 %/yr; an abrupt +15 % shift flagged **94 %** of the time with the
  changepoint dated to a median 55 days (p90 238).
- **Bundled 10-year dataset** (120 gaugings; `auto` selects 1 segment) —
  in-sample R² **0.986**, leave-one-out RMSPE **9.9 %** (bias +0.4 %), and an
  earliest-70 % → latest-30 % temporal hold-out RMSPE of **8.8 %**.

These are estimator-quality figures on synthetic and bundled data, not a
comparison against other software.

## Changelog

**v0.3.1** (current release)

- **`segments="auto"` no longer over-segments a single-control channel when
  `h0` is estimated.** A few-centimetre error in the estimated stage of zero
  flow bends the log–log line, and the forward knot search was "correcting" the
  bend with a spurious breakpoint that BIC then accepted. The single-power-law
  baseline is now re-profiled for `h0` (within ±0.10 m,
  `piecewise.H0_REFINE_WINDOW_M`) before the knots are scored, so it has the
  same freedom a knot implicitly adds. In a Monte-Carlo study (single true
  power law, ~7% noise, n=60) the false-split rate drops from ~48% to ~1%,
  matching the known-`h0` case; two- and three-control recovery and breakpoint
  accuracy are unchanged. A single-segment `auto` fit with an estimated `h0`
  now reports the refined value. Forced `segments=N` and a user-supplied `h0`
  are unaffected.

**v0.3.0**

- **Imposed exponent** – `fit_rating_curve(fixed_b=…)`, `rca fit --exponent`,
  `rca report --exponent`, `run_fit(fixed_b=…)` and an "Impose the exponent b"
  checkbox in the app: pin the power-law exponent from the control type and fit
  only the coefficient, for low-flow / narrow-range records that cannot identify
  `b` themselves. Single-segment OLS; the fit carries `b_fixed` and an
  uncorrelated cloud becomes a "provisional" warning rather than a hard reject.
- **Smarter stuck-gauge detection** – a repeated stage value is only rejected
  when the discharge repeats too, or the readings are a consecutive run of
  visits; a common low-flow stage revisited over the years with different
  discharges is no longer dropped.
- **Actionable rejection message** – a rejected free fit now points at the fixes
  (impose the exponent, or correct the stage column); ambiguous-column messages
  name the override. New `rca validate --stage-column / --discharge-column /
  --date-column`; `clean_measurements_to_csv(column_overrides=…)`.
- **Rebuilt web UI** – a single top-to-bottom column, **no sidebar**: upload → a
  *Detected layout* panel that opens itself only when detection is unsure →
  a *Fit* section with curve shape, method (least-squares / Bayesian), *set h₀ by
  hand* and *impose the exponent b* all on screen, plus *Uncertainty & point
  flags* and *Advanced* drawers → the result (banner + equation + a/b/h₀/R² +
  one context line), the plot, *Diagnostics*, and the downloads.
- **Excel report fixes** – (1) a fractional chart line width was corrupting the
  file (`Removed Part: /xl/drawings/drawing2.xml` on open) — now integer EMU;
  (2) date columns show as `yyyy-mm-dd` and every column is auto-sized (no more
  `######`); (3) the rating-curve / band / Manning charts are XY **scatter**
  charts with visible bottom/left axes, titles and legend each given their own
  margin (openpyxl reserves none), and the residuals chart thins its date axis.
- **Bayesian pins** – `[bayesian]` extra bumped to `ratingcurve>=1.1`,
  `pymc>=5.28,<6` (ratingcurve 1.1 requires pymc ≥ 5.28.1); tested against
  ratingcurve 1.1.0 / pymc 5.28.
- CI runs the suite on Python 3.10–3.13 (plus a pyflakes lint) on every push and
  pull request.

**v0.2.0** — replace the deprecated Streamlit
`use_container_width` with `width="stretch"` (the app extra now needs
`streamlit>=1.49`); `rca app` exits cleanly on Ctrl+C; declare Python 3.10–3.13.

**v0.1.0** — first PyPI release: cleaning/validation, weighted & piecewise fits,
optional Bayesian backend, bootstrap and posterior uncertainty (incl. breakpoint
and changepoint), leave-one-out error, Manning cross-section check, Excel report,
`rca` CLI and Streamlit app.

Releases: <https://github.com/ZergFromZ0rg/Rating-Curve-Automater/releases>
