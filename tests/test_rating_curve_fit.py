import pandas as pd
import numpy as np

from src.rating_curve_fitting import fit_rating_curve


def test_fit_rating_curve_recovers_expected_parameters():
    h0 = 0.18
    a_true = 1.18
    b_true = 1.72

    h = np.linspace(0.22, 1.1, 80)
    q = a_true * (h - h0) ** b_true
    noise = np.linspace(-0.015, 0.02, len(h))
    q_noisy = q * (1 + noise)

    df = pd.DataFrame({
        "Stage Above Bed (m)": h,
        "Measured Discharge Q (m³/s)": q_noisy,
        "is_valid": True,
    })

    result = fit_rating_curve(df)

    assert result["a"] > 0.8
    assert result["a"] < 1.6
    assert result["b"] > 1.4
    assert result["b"] < 2.1
    assert result["r_squared"] > 0.9
