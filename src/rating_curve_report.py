from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.chart import LineChart, Reference
from openpyxl.styles import PatternFill


def build_observed_modeled_table(
    df: pd.DataFrame,
    a: float,
    b: float,
    h0: float,
    uncertainty_threshold: float = 0.25,
) -> pd.DataFrame:
    """Build an observed-vs-modeled table with uncertainty flags."""
    working = df.copy()
    working = working.dropna(subset=["Stage Above Bed (m)", "Measured Discharge Q (m³/s)"]).copy()

    stage = working["Stage Above Bed (m)"].astype(float).to_numpy()
    observed = working["Measured Discharge Q (m³/s)"].astype(float).to_numpy()
    modeled = a * np.power(np.maximum(stage - h0, 1e-9), b)
    residual = observed - modeled
    relative_error = np.abs(residual / np.maximum(observed, 1e-9))

    flags = np.where(relative_error > uncertainty_threshold, "Uncertain", "Normal")

    output = pd.DataFrame({
        "Date": working.get("Date", pd.Series(index=working.index, dtype="object")),
        "Stage Above Bed (m)": stage,
        "Measured Discharge Q (m³/s)": observed,
        "Modeled Discharge Q (m³/s)": modeled,
        "Residual": residual,
        "Relative Error": relative_error,
        "Uncertainty Flag": flags,
    })

    return output


def build_summary_table(fit: dict, table: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        [
            {"Metric": "a", "Value": fit["a"]},
            {"Metric": "b", "Value": fit["b"]},
            {"Metric": "h0", "Value": fit["h0"]},
            {"Metric": "R^2", "Value": fit["r_squared"]},
            {"Metric": "Valid points", "Value": len(table)},
            {"Metric": "Uncertain points", "Value": int((table["Uncertainty Flag"] == "Uncertain").sum())},
            {"Metric": "Normal points", "Value": int((table["Uncertainty Flag"] == "Normal").sum())},
        ]
    )
    return summary


def export_rating_curve_report(
    df: pd.DataFrame,
    output_path: str | Path,
    a: float,
    b: float,
    h0: float,
    uncertainty_threshold: float = 0.25,
) -> Path:
    original_data = df.copy()
    table = build_observed_modeled_table(df, a=a, b=b, h0=h0, uncertainty_threshold=uncertainty_threshold)

    if len(table) > 1:
        ss_res = float(np.sum((table["Measured Discharge Q (m³/s)"] - table["Modeled Discharge Q (m³/s)"]) ** 2))
        ss_tot = float(np.sum((table["Measured Discharge Q (m³/s)"] - table["Measured Discharge Q (m³/s)"].mean()) ** 2))
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
    else:
        r_squared = 1.0

    fit_summary = {
        "a": a,
        "b": b,
        "h0": h0,
        "r_squared": r_squared,
    }
    summary = build_summary_table(fit_summary, table)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        original_data.to_excel(writer, sheet_name="Original Data", index=False)
        table.to_excel(writer, sheet_name="Observed vs Modeled", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)

        plot_data = pd.DataFrame(
            {
                "Stage Above Bed (m)": table["Stage Above Bed (m)"],
                "Observed Discharge Q (m³/s)": table["Measured Discharge Q (m³/s)"],
                "Modeled Discharge Q (m³/s)": table["Modeled Discharge Q (m³/s)"],
                "Uncertain Observed Discharge Q (m³/s)": np.where(table["Uncertainty Flag"] == "Uncertain", table["Measured Discharge Q (m³/s)"], np.nan),
            }
        )
        plot_data.to_excel(writer, sheet_name="Plot Data", index=False)

        workbook = writer.book
        plot_ws = workbook.create_sheet("Plot")
        plot_ws.append(["Stage Above Bed (m)", "Observed Discharge Q (m³/s)", "Modeled Discharge Q (m³/s)", "Uncertain Observed Discharge Q (m³/s)"])
        for row in plot_data.values.tolist():
            plot_ws.append(row)

        chart = LineChart()
        chart.title = "Rating Curve"
        chart.x_axis.title = "Stage Above Bed (m)"
        chart.y_axis.title = "Discharge (m³/s)"
        chart.style = 13
        chart.legend.position = "r"
        chart.height = 9
        chart.width = 18
        chart.y_axis.scaling.min = 0

        observed_data = Reference(plot_ws, min_col=2, max_col=2, min_row=1, max_row=plot_ws.max_row)
        modeled_data = Reference(plot_ws, min_col=3, max_col=3, min_row=1, max_row=plot_ws.max_row)
        uncertain_data = Reference(plot_ws, min_col=4, max_col=4, min_row=1, max_row=plot_ws.max_row)
        categories = Reference(plot_ws, min_col=1, max_col=1, min_row=2, max_row=plot_ws.max_row)

        chart.add_data(observed_data, titles_from_data=True)
        chart.add_data(modeled_data, titles_from_data=True)
        chart.add_data(uncertain_data, titles_from_data=True)
        chart.set_categories(categories)

        chart.series[0].graphicalProperties.line.solidFill = "1F77B4"
        chart.series[0].graphicalProperties.line.width = 2
        chart.series[0].marker.size = 6
        chart.series[0].marker.symbol = "circle"
        chart.series[0].marker.graphicalProperties.solidFill = "1F77B4"

        chart.series[1].graphicalProperties.line.solidFill = "2CA02C"
        chart.series[1].graphicalProperties.line.width = 2
        chart.series[1].marker.size = 4
        chart.series[1].marker.symbol = "triangle"
        chart.series[1].marker.graphicalProperties.solidFill = "2CA02C"

        chart.series[2].graphicalProperties.line.noFill = True
        chart.series[2].marker.size = 7
        chart.series[2].marker.symbol = "diamond"
        chart.series[2].marker.graphicalProperties.solidFill = "D62728"

        plot_ws.add_chart(chart, "F2")

        observed_ws = workbook["Observed vs Modeled"]
        flagged_fill = PatternFill(fill_type="solid", fgColor="FFC000")
        normal_fill = PatternFill(fill_type="solid", fgColor="C6EFCE")
        for row_idx in range(2, observed_ws.max_row + 1):
            flag = observed_ws.cell(row=row_idx, column=7).value
            fill = flagged_fill if flag == "Uncertain" else normal_fill
            for col_idx in range(1, observed_ws.max_column + 1):
                observed_ws.cell(row=row_idx, column=col_idx).fill = fill

    return output


def main() -> None:
    import pandas as pd

    from src.rating_curve_fitting import fit_rating_curve

    input_path = Path(__file__).resolve().parent.parent / "cleaned_measurements.csv"
    output_path = Path(__file__).resolve().parent.parent / "rating_curve_report.xlsx"

    df = pd.read_csv(input_path)
    fit = fit_rating_curve(df)
    export_rating_curve_report(df, output_path, a=fit["a"], b=fit["b"], h0=fit["h0"])

    print(f"Rating-curve report written to: {output_path.name}")


if __name__ == "__main__":
    main()
