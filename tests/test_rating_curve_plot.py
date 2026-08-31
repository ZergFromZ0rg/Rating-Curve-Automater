import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from rating_curve_automater.rating_curve_plot import make_rating_curve_figure


def _frame():
    h = np.linspace(0.25, 1.1, 30)
    q = 1.18 * (h - 0.18) ** 1.72
    return pd.DataFrame(
        {
            "Stage Above Bed (m)": h,
            "Measured Discharge Q (m³/s)": q,
            "is_valid": True,
            "has_warning": [i % 10 == 0 for i in range(len(h))],
        }
    )


def test_make_rating_curve_figure_returns_axes_with_series():
    fig = make_rating_curve_figure(_frame(), a=1.18, b=1.72, h0=0.18)

    ax = fig.axes[0]
    labels = [line.get_label() for line in ax.get_lines()]
    assert any("Q =" in label for label in labels)
    assert ax.get_xlabel() == "Stage above bed (m)"


def test_make_rating_curve_figure_reuses_supplied_figure_and_logscale():
    from matplotlib.figure import Figure

    fig = Figure()
    returned = make_rating_curve_figure(_frame(), a=1.18, b=1.72, h0=0.18, figure=fig, log_scale=True)

    assert returned is fig
    assert fig.axes[0].get_xscale() == "log"
