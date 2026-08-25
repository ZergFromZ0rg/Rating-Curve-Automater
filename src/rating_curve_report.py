from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_observed_modeled_table(
    df: pd.DataFrame,
    a: float,
    b: float,
    h0: float,
    uncertainty_threshold: float = 0.25,
) -> pd.DataFrame:
    """Build an observed-vs-modeled table with uncertainty flags.

    The flagging rule is based on the relative residual between observed and modeled discharge.
    """
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


def export_rating_curve_report(
    df: pd.DataFrame,
    output_path: str | Path,
    a: float,
    b: float,
    h0: float,
    uncertainty_threshold: float = 0.25,
) -> Path:
    table = build_observed_modeled_table(df, a=a, b=b, h0=h0, uncertainty_threshold=uncertainty_threshold)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="Observed vs Modeled", index=False)
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
