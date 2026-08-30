"""Headless controller for the rating-curve workflow.

All of the load -> validate -> fit -> export logic lives here as plain
functions/methods that return data, with no Tkinter involved, so the workflow
can be exercised directly in tests. ``rating_curve_app.py`` is a thin view on
top of :class:`RatingCurveWorkflow`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.field_measurement_validation import clean_and_validate_measurements
from src.loader import LoadReport, load_measurements
from src.rating_curve_fitting import fit_rating_curve
from src.rating_curve_report import export_rating_curve_report
from src.schema import DATE, DISCHARGE_CMS, FIELD_LABELS, STAGE_M

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
    flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_blocking_issues(self) -> bool:
        return self.invalid_count > 0

    def summary_line(self) -> str:
        return (
            f"Valid rows: {self.valid_count}. Invalid rows: {self.invalid_count}. "
            f"Warnings: {self.warning_count}."
        )


@dataclass
class FitOutcome:
    params: dict

    def summary_line(self) -> str:
        p = self.params
        source = "estimated" if p["h0_estimated"] else "user-specified"
        if p.get("is_segmented"):
            head = f"Segmented at H = {p['breakpoint']:.3f} m  |  {p['equation']}"
        else:
            head = f"Q = {p['a']:.6f} * (H - {p['h0']:.3f})^{p['b']:.6f}"
        return f"{head}  | R² = {p['r_squared']:.4f} | h0 {source} | {p['n_points']} points"


class RatingCurveWorkflow:
    def __init__(self) -> None:
        self.cleaned_df: pd.DataFrame | None = None
        self.fit_params: dict | None = None

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
        self.fit_params = None

        invalid_rows = cleaned.loc[~cleaned["is_valid"]]
        warning_rows = cleaned.loc[cleaned["has_warning"]]

        return ValidationResult(
            cleaned=cleaned,
            valid_count=int(cleaned["is_valid"].sum()),
            invalid_count=int((~cleaned["is_valid"]).sum()),
            warning_count=int(cleaned["has_warning"].sum()),
            load_report=report,
            flags=_describe_rows(invalid_rows, "validation_notes"),
            warnings=_describe_rows(warning_rows, "warning_notes"),
        )

    def run_fit(self, h0: float | None = None, segments: int = 1) -> FitOutcome:
        if self.cleaned_df is None:
            raise RuntimeError("Validate a dataset before fitting.")
        params = fit_rating_curve(self.cleaned_df, h0=h0, segments=segments)
        self.fit_params = params
        return FitOutcome(params=params)

    def export_report(
        self,
        output_path: str | Path,
        uncertainty_threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD,
    ) -> Path:
        if self.cleaned_df is None or self.fit_params is None:
            raise RuntimeError("Complete validation and fitting before exporting.")
        return export_rating_curve_report(
            self.cleaned_df,
            output_path,
            a=self.fit_params["a"],
            b=self.fit_params["b"],
            h0=self.fit_params["h0"],
            uncertainty_threshold=uncertainty_threshold,
            r_squared=self.fit_params["r_squared"],
            fit=self.fit_params,
        )
