# Benchmarks

Reproducible validation of the estimator. Everything is seeded — a rerun on the
same package version reproduces the numbers.

```bash
python3 benchmarks/simulation_study.py            # full run (~5-10 min)
python3 benchmarks/simulation_study.py --quick     # fewer trials (~1 min)
python3 benchmarks/simulation_study.py --only a d  # subset
```

Output: Markdown tables on stdout + `benchmarks/results/simulation_study.json`.

## `simulation_study.py`

Four studies. A-C are Monte-Carlo: build a stage-discharge curve with **known**
parameters, add unbiased multiplicative lognormal measurement noise, fit it
through the same `fit_rating_curve` path a user runs, and score the result.

| Study | Question | Key numbers |
|-------|----------|-------------|
| **A — single power law** | Does the weighted log-log fit recover a known exponent, and are the bootstrap intervals calibrated? | bias / RMSE of `b`; empirical coverage of the 95% CIs on `b`, `a`, `h0`; pointwise coverage of the 95% confidence band; median band half-width. Plus a 3×3 sweep over gauging count (15/30/60) and noise level (5/10/20 %). |
| **B — piecewise + BIC** | Does BIC pick the right number of segments, and land the breakpoints? | P(correct segment count) for 1-, 2-, 3-segment truth; false-split rate on single-control data; breakpoint error as % of gauged stage range. Run twice — `h0` supplied vs. estimated — to isolate point-of-zero-flow error, which bends the log-log line either way and once let BIC "correct" the bend with a spurious knot (~48 % false-split with an estimated `h0`). v0.3.1 re-profiles `h0` for the single-power-law baseline before scoring knots; the false-split rate is now ~1 %, matching the `h0`-supplied case. |
| **C — drift detection** | Does the temporal-drift test fire at the right trend magnitudes without false alarms, and locate a step change? | detection rate vs. true trend (0/2/4/6 %/yr); false-positive rate at 0 %/yr; recovered trend; for a +15 % step at a known date: flag rate, changepoint-date error (median / p90 days). |
| **D — real-data CV** | Out-of-sample accuracy on the bundled 10-year record. | leave-one-out RMSPE / MAE / bias; earliest-70 % → latest-30 % temporal hold-out RMSPE; for single and BIC-`auto` segmentation. |

### Notes on the synthetic model

- Noise is `q_true · lognormal(-s²/2, s)` with `s` = the stated CV, so the
  multiplier has mean 1 (no discharge bias injected).
- Stages are log-spaced in `H - h0` then jittered — gaugings cluster at low flow
  with a thin high-flow tail, like a real record.
- Study C shuffles the stage/date pairing so stage is not confounded with time
  (an unconfounded gauging schedule); the confounded case is what the estimator
  reports as `flag = "unassessable"`.
- Piecewise truth curves are generated in the same log-space linear-spline
  parameterisation the fitter uses, so "correct breakpoint" is well defined.

The numbers are estimator-quality figures, not a comparison against other
software. The only peer tool solving the same problem that installs as a library
is [`thodson-usgs/ratingcurve`](https://github.com/thodson-usgs/ratingcurve),
available here as the `[bayesian]` backend (`method="bayesian"`).
