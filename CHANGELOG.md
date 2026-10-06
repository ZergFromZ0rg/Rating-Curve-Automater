# Changelog

## v0.3.10 — 2026-10-05

- `rca app` now launches with the light theme (it was forcing Streamlit's dark
  theme onto the light-styled page, and the repo's `.streamlit/config.toml`
  isn't shipped in the wheel).
- Removed the hand-positioned header overlay on the rating-table grid; it
  hard-coded "95%" and fixed pixel offsets and could drift out of alignment.
- The a/b bootstrap intervals are now labelled one-sided when h0 is pinned at
  its 0 m bound (the point estimate otherwise sits exactly on the interval edge).
- CI/publish workflows moved to Node 24 action versions (checkout v5,
  setup-python v6, artifact v5).

## v0.3.9 — 2026-10-02

- Rebuilt the Streamlit dashboard as a compact, responsive workspace with the
  fit controls and summary arranged around the rating-curve graph.
- Added a dedicated detailed-results screen for the rating table, residuals,
  fit details, and validation checks.
- Refined the light visual theme, spacing, typography, tables, controls, and
  graph fullscreen behavior for a consistent interface.
- Restyled the rating-curve and residual plots for clearer publication-style
  lines, points, uncertainty bands, axes, and labels.
- Improved the data-column review status and expanded automated coverage for
  the redesigned summary and download workflow.

## v0.3.1 — 2026-09-06

- Improved `segments="auto"` behavior when estimating `h0`.
- Added imposed-exponent fitting and smarter stuck-gauge detection.
- Rebuilt the Streamlit interface around a single reviewable workflow.
- Fixed Excel chart, date-formatting, column-sizing, and Bayesian dependency issues.

## v0.3.0

- Added imposed-exponent fitting, improved validation messages, and column overrides.
- Added the rebuilt web UI and Excel report fixes.
- Added CI coverage for Python 3.10–3.13.

## v0.2.0

- Updated Streamlit compatibility and CLI shutdown behavior.
- Declared support for Python 3.10–3.13.

## v0.1.0

- Initial PyPI release with cleaning, validation, weighted and piecewise fits,
  Bayesian support, uncertainty bands, diagnostics, reports, CLI, and web UI.
