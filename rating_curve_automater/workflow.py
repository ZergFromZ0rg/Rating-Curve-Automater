"""Headless controller for the rating-curve workflow.

All of the load -> validate -> fit -> export logic lives here as plain
functions/methods that return data, with no UI involved, so the workflow can be
exercised directly in tests. ``app.py`` (Streamlit) is a thin view on top of
:class:`RatingCurveWorkflow`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from rating_curve_automater.field_measurement_validation import clean_and_validate_measurements
from rating_curve_automater.loader import LoadReport, load_measurements
from rating_curve_automater.rating_curve_fitting import DEFAULT_DISCHARGE_UNCERTAINTY_PCT, fit_rating_curve
from rating_curve_automater.rating_curve_report import export_rating_curve_report
from rating_curve_automater.rating_curve_uncertainty import DEFAULT_N_BOOTSTRAP
from rating_curve_automater.rating_table import DEFAULT_STAGE_STEP_M, build_rating_table
from rating_curve_automater.rating_table import export_rating_table_csv as _write_rating_table_csv
from rating_curve_automater.schema import DATE, DISCHARGE_CMS, FIELD_LABELS, SITE, STAGE_M

DEFAULT_SHEET_NAME: str | None = None
DEFAULT_UNCERTAINTY_THRESHOLD = 0.25

_DISPLAY_COLS = [DATE, STAGE_M, DISCHARGE_CMS]


def _describe_rows(df: pd.DataFrame, note_col: str) -> list[str]:
    lines = []
    for _, row in df.iterrows():
        parts = []
        for col in _DISPLAY_COLS:
            if col in df.columns:
                parts.append(f"{FIELD_LABELS[col]}: {row[col]}")
        parts.append(str(row[note_col]))
        lines.append(" | ".join(parts))
    return lines


@dataclass
class ValidationResult:
    cleaned: pd.DataFrame
    valid_count: int
    invalid_count: int
    warning_count: int
    load_report: LoadReport | None = None
    sites: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_blocking_issues(self) -> bool:
        return self.invalid_count > 0

    @property
    def is_multi_site(self) -> bool:
        return len(self.sites) > 1

    def summary_line(self) -> str:
        return (
            f"Valid rows: {self.valid_count}. Invalid rows: {self.invalid_count}. "
            f"Warnings: {self.warning_count}."
        )


@dataclass
class FitOutcome:
    params: dict
    site: str | None = None

    @property
    def warnings(self) -> list[str]:
        return list(self.params.get("warnings", []))

    @property
    def is_plausible(self) -> bool:
        return self.params.get("is_plausible", True)

    @property
    def bands(self) -> dict | None:
        return self.params.get("bands")

    @property
    def drift(self) -> dict | None:
        return self.params.get("drift")

    def summary_line(self) -> str:
        p = self.params
        source = "estimated" if p["h0_estimated"] else "user-specified"
        if p.get("is_segmented"):
            bps = p.get("breakpoints", [p.get("breakpoint")])
            head = (
                f"{p.get('n_segments', len(bps) + 1)} segments, breaks at "
                f"{', '.join(f'{b:.3f}' for b in bps)} m  |  {p['equation']}"
            )
        else:
            head = f"Q = {p['a']:.6f} * (H - {p['h0']:.3f})^{p['b']:.6f}"
        prefix = f"[{self.site}] " if self.site else ""
        estimator = "Bayesian" if p.get("method") == "bayesian" else "least squares"
        line = f"{prefix}{head}  | {estimator} | R² = {p['r_squared']:.4f} | h0 {source} | {p['n_points']} points"
        if p.get("weighted"):
            line += " | weighted by per-point uncertainty"
        elif p.get("uncertainty_source") == "column":
            line += " | per-point uncertainty (uniform)"
        bands = p.get("bands")
        if bands and bands.get("b_ci"):
            lo, hi = bands["b_ci"]
            pct = int(round(bands["level"] * 100))
            line += f" | {pct}% CI b∈[{lo:.2f}, {hi:.2f}]"
        elif bands:
            pct = int(round(bands["level"] * 100))
            line += f" | {pct}% band ±{bands['ci_halfwidth_pct_at_median']:.0f}% at median stage"
        if self.warnings:
            line = "⚠ " + line + "\n   " + "\n   ".join(self.warnings)
        drift = p.get("drift")
        if drift and drift["flag"] != "none":
            line += f"\n   ⏳ {drift['message']}"
        return line


def _sites_in(df: pd.DataFrame) -> list[str]:
    if SITE not in df.columns:
        return []
    values = df.loc[df["is_valid"], SITE].dropna().astype(str).str.strip()
    values = values[values != ""]
    return sorted(values.unique().tolist())


class RatingCurveWorkflow:
    def __init__(self) -> None:
        self.cleaned_df: pd.DataFrame | None = None
        self.fit_df: pd.DataFrame | None = None
        self.fit_params: dict | None = None
        self.selected_site: str | None = None

    def load_and_validate(
        self,
        dataset_path: str | Path,
        sheet_name: str | int | None = DEFAULT_SHEET_NAME,
        column_overrides: dict[str, str] | None = None,
        header_row: int | None = None,
    ) -> ValidationResult:
        canonical, report = load_measurements(
            dataset_path,
            sheet=sheet_name,
            header_row=header_row,
            column_overrides=column_overrides,
        )
        if not report.ok:
            missing = ", ".join(report.mapping.unresolved_required)
            raise ValueError(
                f"Could not identify required column(s): {missing}.\n{report.describe()}"
            )
        cleaned = clean_and_validate_measurements(canonical)
        self.cleaned_df = cleaned
        self.fit_df = None
        self.fit_params = None
        self.selected_site = None

        invalid_rows = cleaned.loc[~cleaned["is_valid"]]
        warning_rows = cleaned.loc[cleaned["has_warning"]]

        return ValidationResult(
            cleaned=cleaned,
            valid_count=int(cleaned["is_valid"].sum()),
            invalid_count=int((~cleaned["is_valid"]).sum()),
            warning_count=int(cleaned["has_warning"].sum()),
            load_report=report,
            sites=_sites_in(cleaned),
            flags=_describe_rows(invalid_rows, "validation_notes"),
            warnings=_describe_rows(warning_rows, "warning_notes"),
        )

    def run_fit(
        self,
        h0: float | None = None,
        segments: int | str = 1,
        site: str | None = None,
        discharge_uncertainty_pct: float = DEFAULT_DISCHARGE_UNCERTAINTY_PCT,
        n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
        random_state: int | None = 0,
        method: str = "ols",
    ) -> FitOutcome:
        if self.cleaned_df is None:
            raise RuntimeError("Validate a dataset before fitting.")

        fit_df = self.cleaned_df
        if site is not None:
            if SITE not in fit_df.columns:
                raise ValueError("The dataset has no site column to filter on.")
            fit_df = fit_df[fit_df[SITE].astype(str).str.strip() == site]
            if fit_df.empty:
                raise ValueError(f"No rows for site '{site}'.")

        params = fit_rating_curve(
            fit_df,
            h0=h0,
            segments=segments,
            discharge_uncertainty_pct=discharge_uncertainty_pct,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            method=method,
        )
        self.fit_df = fit_df
        self.fit_params = params
        self.selected_site = site
        return FitOutcome(params=params, site=site)

    def export_report(
        self,
        output_path: str | Path,
        uncertainty_threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD,
        rating_table_step: float = DEFAULT_STAGE_STEP_M,
    ) -> Path:
        if self.fit_df is None or self.fit_params is None:
            raise RuntimeError("Complete validation and fitting before exporting.")
        return export_rating_curve_report(
            self.fit_df,
            output_path,
            a=self.fit_params["a"],
            b=self.fit_params["b"],
            h0=self.fit_params["h0"],
            uncertainty_threshold=uncertainty_threshold,
            r_squared=self.fit_params["r_squared"],
            fit=self.fit_params,
            site=self.selected_site,
            rating_table_step=rating_table_step,
        )

    def rating_table(
        self,
        step: float = DEFAULT_STAGE_STEP_M,
        stage_min: float | None = None,
        stage_max: float | None = None,
    ) -> pd.DataFrame:
        """Stage → discharge lookup table for the current fit (see
        :func:`rating_curve_automater.rating_table.build_rating_table`)."""
        if self.fit_params is None:
            raise RuntimeError("Fit a curve before building the rating table.")
        return build_rating_table(
            self.fit_params, step=step, stage_min=stage_min, stage_max=stage_max
        )

    def export_rating_table_csv(
        self,
        output_path: str | Path,
        step: float = DEFAULT_STAGE_STEP_M,
        stage_min: float | None = None,
        stage_max: float | None = None,
    ) -> Path:
        if self.fit_params is None:
            raise RuntimeError("Fit a curve before exporting the rating table.")
        return _write_rating_table_csv(
            self.fit_params, output_path, step=step, stage_min=stage_min, stage_max=stage_max
        )
