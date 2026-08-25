from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_H0 = 0.18


def fit_rating_curve(df: pd.DataFrame, h0: float | None = None) -> dict:
    """Fit a simple power-law rating curve: Q = a * (H - h0)^b.

    The model is fitted on valid measurement rows in the dataframe.
    """
    working = df.copy()

    if h0 is None:
        h0 = DEFAULT_H0

    if "Stage Above Bed (m)" not in working.columns:
        raise ValueError("Stage column 'Stage Above Bed (m)' is required.")
    if "Measured Discharge Q (m³/s)" not in working.columns:
        raise ValueError("Discharge column 'Measured Discharge Q (m³/s)' is required.")

    working = working[working.get("is_valid", True)].copy()
    working = working.dropna(subset=["Stage Above Bed (m)", "Measured Discharge Q (m³/s)"]).copy()

    stage = working["Stage Above Bed (m)"].to_numpy(dtype=float)
    discharge = working["Measured Discharge Q (m³/s)"].to_numpy(dtype=float)

    x = stage - h0
    valid = x > 0
    stage = stage[valid]
    discharge = discharge[valid]
    x = x[valid]

    if len(x) < 2:
        raise ValueError("Not enough valid stage-discharge points to fit a rating curve.")

    log_x = np.log(x)
    log_q = np.log(discharge)

    slope, intercept = np.polyfit(log_x, log_q, 1)
    a = np.exp(intercept)
    b = slope

    predicted = a * np.power(stage - h0, b)
    residuals = discharge - predicted
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((discharge - np.mean(discharge)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot != 0 else 1.0

    return {
        "a": float(a),
        "b": float(b),
        "h0": float(h0),
        "r_squared": float(r_squared),
        "equation": f"Q = {a:.6f} * (H - {h0:.3f})^{b:.6f}",
    }


def main() -> None:
    csv_path = Path(__file__).resolve().parent.parent / "cleaned_measurements.csv"
    df = pd.read_csv(csv_path)
    fit = fit_rating_curve(df)

    print("Rating curve fit results")
    print(f"a = {fit['a']:.6f}")
    print(f"b = {fit['b']:.6f}")
    print(f"h0 = {fit['h0']:.3f}")
    print(f"R^2 = {fit['r_squared']:.4f}")
    print(f"Equation: {fit['equation']}")


if __name__ == "__main__":
    main()
