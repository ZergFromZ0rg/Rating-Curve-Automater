"""Monte-Carlo validation of the rating-curve estimator.

Everything here is synthetic: we build stage-discharge curves with *known*
parameters, add realistic multiplicative measurement noise, fit them with the
same code path a user runs, and measure how well the truth is recovered and how
well the reported uncertainty is calibrated. Plus one cross-validation pass on
the bundled real dataset.

Run:  python3 benchmarks/simulation_study.py            (full, ~5-10 min)
      python3 benchmarks/simulation_study.py --quick    (fewer trials, ~1 min)

Results print as Markdown tables and are written to
benchmarks/results/simulation_study.json. Seeded — reruns reproduce the numbers.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from rating_curve_automater import __version__
from rating_curve_automater.field_measurement_validation import (
    DEFAULT_DATASET,
    DEFAULT_DATASET_SHEET,
)
from rating_curve_automater.rating_curve_fitting import (
    fit_rating_curve,
    leave_one_out_error,
    predict_discharge,
)
from rating_curve_automater.schema import DATE, DISCHARGE_CMS, STAGE_M
from rating_curve_automater.workflow import RatingCurveWorkflow

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CI_LEVEL = 0.95


# --------------------------------------------------------------------------- #
# synthetic-data helpers
# --------------------------------------------------------------------------- #
def _noisy(q_true: np.ndarray, cv: float, rng: np.random.Generator) -> np.ndarray:
    """Unbiased multiplicative lognormal noise with coefficient of variation ~cv."""
    s = float(cv)
    return q_true * rng.lognormal(mean=-0.5 * s * s, sigma=s, size=q_true.shape)


def _sample_stages(h0: float, hi_above: float, n: int, rng: np.random.Generator,
                   lo_above: float = 0.12) -> np.ndarray:
    """n stages between h0+lo_above and h0+hi_above, log-spaced in (H-h0) then
    jittered — mimics gaugings clustered at low flow with a thin high-flow tail."""
    u = rng.uniform(np.log(lo_above), np.log(hi_above), n)
    return np.sort(h0 + np.exp(u))


def _single_curve(h: np.ndarray, a: float, b: float, h0: float) -> np.ndarray:
    return a * np.power(h - h0, b)


def _piecewise_curve(h: np.ndarray, h0: float, a1: float, slopes: list[float],
                     breaks: list[float]) -> np.ndarray:
    """Continuous piecewise power law, built in log space (same model the fitter
    uses). slopes has len(breaks)+1 entries."""
    u = np.log(h - h0)
    knots = [np.log(bp - h0) for bp in breaks]
    y = np.log(a1) + slopes[0] * u
    for j, k in enumerate(knots):
        y = y + (slopes[j + 1] - slopes[j]) * np.maximum(u - k, 0.0)
    return np.exp(y)


def _dates(n: int, span_days: int, rng: np.random.Generator) -> list[date]:
    start = date(2015, 1, 1)
    offs = np.sort(rng.integers(0, span_days, n))
    return [start + timedelta(days=int(o)) for o in offs]


def _pct(x: float) -> str:
    return f"{x:.1f}"


# --------------------------------------------------------------------------- #
# Study A — single power law: parameter recovery + interval calibration
# --------------------------------------------------------------------------- #
def study_a(quick: bool) -> dict:
    print("\n=== Study A: single power-law recovery + interval calibration ===")
    a_true, b_true, h0_true = 2.2, 1.90, 0.15
    hi_above = 2.5

    def one_config(n: int, cv: float, n_trials: int, n_boot: int) -> dict:
        b_rel, h0_abs = [], []
        b_hit = h0_hit = a_hit = band_pts = band_tot = full_cover = n_ok = 0
        hw = []
        for i in range(n_trials):
            rng = np.random.default_rng(1_000 + i)
            h = _sample_stages(h0_true, hi_above, n, rng)
            q = _noisy(_single_curve(h, a_true, b_true, h0_true), cv, rng)
            df = pd.DataFrame({STAGE_M: h, DISCHARGE_CMS: q})
            try:
                fit = fit_rating_curve(df, segments=1, n_bootstrap=n_boot,
                                       ci_level=CI_LEVEL, random_state=i,
                                       _diagnostics=False)
            except Exception:
                continue
            n_ok += 1
            b_rel.append((fit["b"] - b_true) / b_true)
            h0_abs.append(fit["h0"] - h0_true)
            bands = fit.get("bands")
            if not bands:
                continue
            if bands.get("b_ci"):
                lo, hi = bands["b_ci"]
                b_hit += lo <= b_true <= hi
            if bands.get("a_ci"):
                lo, hi = bands["a_ci"]
                a_hit += lo <= a_true <= hi
            if bands.get("h0_ci"):
                lo, hi = bands["h0_ci"]
                h0_hit += lo <= h0_true <= hi
            grid = bands["stage"]
            true_q = _single_curve(grid, a_true, b_true, h0_true)
            inside = (true_q >= bands["ci_lower"]) & (true_q <= bands["ci_upper"])
            band_pts += int(inside.sum())
            band_tot += inside.size
            full_cover += int(inside.all())
            hw.append(bands["ci_halfwidth_pct_at_median"])

        b_rel = np.asarray(b_rel) * 100.0
        h0_abs = np.asarray(h0_abs)
        return {
            "n_gaugings": n, "noise_cv_pct": round(cv * 100, 1),
            "trials": n_ok, "n_bootstrap": n_boot,
            "b_bias_pct": float(np.mean(b_rel)),
            "b_rmse_pct": float(np.sqrt(np.mean(b_rel ** 2))),
            "b_mae_pct": float(np.mean(np.abs(b_rel))),
            "h0_bias_m": float(np.mean(h0_abs)),
            "h0_rmse_m": float(np.sqrt(np.mean(h0_abs ** 2))),
            "b_ci_coverage_pct": 100.0 * b_hit / max(n_ok, 1),
            "a_ci_coverage_pct": 100.0 * a_hit / max(n_ok, 1),
            "h0_ci_coverage_pct": 100.0 * h0_hit / max(n_ok, 1),
            "conf_band_pointwise_coverage_pct": 100.0 * band_pts / max(band_tot, 1),
            "conf_band_full_curve_coverage_pct": 100.0 * full_cover / max(n_ok, 1),
            "median_conf_band_halfwidth_pct": float(np.median(hw)) if hw else float("nan"),
            "target_coverage_pct": CI_LEVEL * 100,
        }

    headline = one_config(n=25, cv=0.08,
                          n_trials=150 if quick else 400,
                          n_boot=250 if quick else 350)
    print(f"  headline (n=25, noise ~8%, {headline['trials']} trials):")
    print(f"    exponent b: bias {_pct(headline['b_bias_pct'])}%  "
          f"RMSE {_pct(headline['b_rmse_pct'])}%  MAE {_pct(headline['b_mae_pct'])}%")
    print(f"    h0: bias {headline['h0_bias_m']:+.3f} m  RMSE {headline['h0_rmse_m']:.3f} m")
    print(f"    95% CI empirical coverage — b {_pct(headline['b_ci_coverage_pct'])}%  "
          f"h0 {_pct(headline['h0_ci_coverage_pct'])}%  a {_pct(headline['a_ci_coverage_pct'])}%")
    print(f"    95% confidence band: pointwise coverage "
          f"{_pct(headline['conf_band_pointwise_coverage_pct'])}%, "
          f"whole-curve {_pct(headline['conf_band_full_curve_coverage_pct'])}%, "
          f"median half-width ±{_pct(headline['median_conf_band_halfwidth_pct'])}%")

    sweep = []
    n_trials = 60 if quick else 120
    n_boot = 150 if quick else 200
    for n in (15, 30, 60):
        for cv in (0.05, 0.10, 0.20):
            row = one_config(n=n, cv=cv, n_trials=n_trials, n_boot=n_boot)
            sweep.append(row)
            print(f"  sweep n={n:>2} cv={cv*100:>4.0f}%:  "
                  f"b RMSE {_pct(row['b_rmse_pct'])}%  "
                  f"b-CI cov {_pct(row['b_ci_coverage_pct'])}%  "
                  f"band cov {_pct(row['conf_band_pointwise_coverage_pct'])}%")
    return {"headline": headline, "sweep": sweep}


# --------------------------------------------------------------------------- #
# Study B — piecewise fit + BIC segment-count selection
# --------------------------------------------------------------------------- #
def study_b(quick: bool) -> dict:
    print("\n=== Study B: piecewise regression + BIC segment selection ===")
    h0 = 0.15
    n_pts = 60
    cv = 0.07
    n_trials = 120 if quick else 250
    out = {}

    def run(name: str, true_k: int, gen, h0_known: bool) -> dict:
        counts = np.zeros(6, dtype=int)
        bp_err_all, bp_err_matched = [], []
        for i in range(n_trials):
            rng = np.random.default_rng(5_000 + i)
            h, true_breaks = gen(rng)
            q = _noisy(h["q_true"], cv, rng)
            df = pd.DataFrame({STAGE_M: h["stage"], DISCHARGE_CMS: q})
            try:
                fit = fit_rating_curve(df, segments="auto",
                                       h0=(h0 if h0_known else None),
                                       n_bootstrap=0, random_state=i,
                                       _diagnostics=False)
            except Exception:
                continue
            k = int(fit.get("n_segments", 1))
            counts[min(k, 5)] += 1
            if true_breaks:
                got = sorted(fit.get("breakpoints", []))
                rng_span = float(h["stage"].max() - h["stage"].min())
                if len(got) == len(true_breaks):
                    err = np.abs(np.array(got) - np.array(sorted(true_breaks)))
                    bp_err_matched.append(float(np.mean(err) / rng_span * 100))
                # nearest-match error regardless of count
                for tb in true_breaks:
                    if got:
                        bp_err_all.append(
                            float(min(abs(g - tb) for g in got) / rng_span * 100))
        total = int(counts.sum())
        res = {
            "case": name, "true_segments": true_k,
            "h0": "known" if h0_known else "estimated", "trials": total,
            "selected_1": int(counts[1]), "selected_2": int(counts[2]),
            "selected_3": int(counts[3]), "selected_4plus": int(counts[4] + counts[5]),
            "correct_k_pct": 100.0 * counts[min(true_k, 5)] / max(total, 1),
        }
        if true_k >= 2:
            res["at_least_2_seg_pct"] = 100.0 * counts[2:].sum() / max(total, 1)
        if bp_err_matched:
            res["breakpoint_err_pct_of_range_when_k_correct"] = float(np.median(bp_err_matched))
        if bp_err_all:
            res["breakpoint_nearest_err_pct_of_range"] = float(np.median(bp_err_all))
        return res

    def gen_1seg(rng):
        h = _sample_stages(h0, 3.0, n_pts, rng)
        return {"stage": h, "q_true": _single_curve(h, 2.0, 1.8, h0)}, []

    def gen_2seg(rng):
        h = _sample_stages(h0, 3.2, n_pts, rng)
        bp = float(np.exp(rng.uniform(np.log(h0 + 0.5), np.log(h0 + 1.4))))
        q = _piecewise_curve(h, h0, a1=2.0, slopes=[1.45, 2.45], breaks=[bp])
        return {"stage": h, "q_true": q}, [bp]

    def gen_3seg(rng):
        h = _sample_stages(h0, 3.6, max(n_pts, 75), rng)
        b1 = float(np.exp(rng.uniform(np.log(h0 + 0.35), np.log(h0 + 0.7))))
        b2 = float(np.exp(rng.uniform(np.log(h0 + 1.1), np.log(h0 + 2.0))))
        q = _piecewise_curve(h, h0, a1=2.0, slopes=[1.3, 2.0, 2.8], breaks=[b1, b2])
        return {"stage": h, "q_true": q}, [b1, b2]

    for h0_known in (True, False):
        tag = "h0 known" if h0_known else "h0 estimated"
        suffix = "_h0known" if h0_known else "_h0est"
        out["one_segment" + suffix] = run("1-segment truth", 1, gen_1seg, h0_known)
        r = out["one_segment" + suffix]
        print(f"  [{tag}] 1-seg truth: picks 1 in {_pct(r['correct_k_pct'])}% "
              f"(false-split rate {_pct(100 - r['correct_k_pct'])}%)")
        out["two_segment" + suffix] = run("2-segment truth", 2, gen_2seg, h0_known)
        r = out["two_segment" + suffix]
        print(f"  [{tag}] 2-seg truth: exact k in {_pct(r['correct_k_pct'])}%, "
              f">=2 seg in {_pct(r['at_least_2_seg_pct'])}%; "
              f"breakpoint err {_pct(r.get('breakpoint_err_pct_of_range_when_k_correct', float('nan')))}% "
              f"of gauged range")
        out["three_segment" + suffix] = run("3-segment truth", 3, gen_3seg, h0_known)
        r = out["three_segment" + suffix]
        print(f"  [{tag}] 3-seg truth: exact k in {_pct(r['correct_k_pct'])}%, "
              f">=2 seg in {_pct(r['at_least_2_seg_pct'])}%; "
              f"breakpoint nearest-match err "
              f"{_pct(r.get('breakpoint_nearest_err_pct_of_range', float('nan')))}% of range")
    return out


# --------------------------------------------------------------------------- #
# Study C — temporal-drift detection
# --------------------------------------------------------------------------- #
def study_c(quick: bool) -> dict:
    print("\n=== Study C: temporal-drift detection ===")
    a_true, b_true, h0 = 2.0, 1.9, 0.15
    n_pts = 40
    span_days = 5 * 365
    cv = 0.07
    n_trials = 80 if quick else 150
    out = {"trend": [], "step": {}}

    for rate in (0.0, 0.02, 0.04, 0.06):
        flags = {"none": 0, "possible": 0, "likely": 0, "unassessable": 0}
        trend_est = []
        n_ok = 0
        for i in range(n_trials):
            rng = np.random.default_rng(9_000 + int(rate * 1000) * 997 + i)
            h = _sample_stages(h0, 2.5, n_pts, rng)
            rng.shuffle(h)  # break stage<->date ordering (unconfounded schedule)
            dts = _dates(n_pts, span_days, rng)
            t_years = np.array([(d - dts[0]).days for d in dts]) / 365.25
            q_true = _single_curve(np.sort(h), a_true, b_true, h0)
            # re-pair sorted-stage curve back to the shuffled order
            order = np.argsort(h)
            q_curve = np.empty_like(q_true)
            q_curve[order] = q_true
            q = _noisy(q_curve * np.exp(rate * t_years), cv, rng)
            df = pd.DataFrame({DATE: [d.isoformat() for d in dts],
                               STAGE_M: h, DISCHARGE_CMS: q})
            try:
                fit = fit_rating_curve(df, segments=1, n_bootstrap=0, random_state=i)
            except Exception:
                continue
            n_ok += 1
            drift = fit.get("drift") or {}
            flags[drift.get("flag", "none")] = flags.get(drift.get("flag", "none"), 0) + 1
            if "trend_pct_per_year" in drift:
                trend_est.append(drift["trend_pct_per_year"])
        detected = flags["possible"] + flags["likely"]
        row = {
            "true_trend_pct_per_year": rate * 100, "trials": n_ok,
            "flag_none_pct": 100.0 * flags["none"] / max(n_ok, 1),
            "flag_possible_pct": 100.0 * flags["possible"] / max(n_ok, 1),
            "flag_likely_pct": 100.0 * flags["likely"] / max(n_ok, 1),
            "flag_unassessable_pct": 100.0 * flags["unassessable"] / max(n_ok, 1),
            "detected_pct": 100.0 * detected / max(n_ok, 1),
            "recovered_trend_median_pct_per_year": float(np.median(trend_est)) if trend_est else float("nan"),
        }
        out["trend"].append(row)
        label = "false-positive rate" if rate == 0 else "detection rate"
        print(f"  {rate*100:.0f}%/yr trend: {label} "
              f"{_pct(row['detected_pct'])}% (likely {_pct(row['flag_likely_pct'])}%); "
              f"recovered {_pct(row['recovered_trend_median_pct_per_year'])}%/yr")

    # step change of known size at a known date
    step_pct = 0.15
    n_step = 60 if quick else 120
    cp_detected = 0
    date_err_days = []
    n_ok = 0
    for i in range(n_step):
        rng = np.random.default_rng(20_000 + i)
        h = _sample_stages(h0, 2.5, n_pts, rng)
        rng.shuffle(h)
        dts = _dates(n_pts, span_days, rng)
        break_at = dts[0] + timedelta(days=span_days // 2)
        order = np.argsort(h)
        q_true = _single_curve(np.sort(h), a_true, b_true, h0)
        q_curve = np.empty_like(q_true)
        q_curve[order] = q_true
        mult = np.array([1.0 + step_pct if d >= break_at else 1.0 for d in dts])
        q = _noisy(q_curve * mult, cv, rng)
        df = pd.DataFrame({DATE: [d.isoformat() for d in dts],
                           STAGE_M: h, DISCHARGE_CMS: q})
        try:
            fit = fit_rating_curve(df, segments=1, n_bootstrap=0, random_state=i)
        except Exception:
            continue
        n_ok += 1
        drift = fit.get("drift") or {}
        cp = drift.get("changepoint")
        if drift.get("flag") in ("possible", "likely"):
            cp_detected += 1
        if cp:
            got = date.fromisoformat(cp["date"])
            date_err_days.append(abs((got - break_at).days))
    out["step"] = {
        "true_step_pct": step_pct * 100, "true_break_at": "record midpoint",
        "trials": n_ok,
        "shift_flagged_pct": 100.0 * cp_detected / max(n_ok, 1),
        "changepoint_located_pct": 100.0 * len(date_err_days) / max(n_ok, 1),
        "changepoint_date_err_days_median": float(np.median(date_err_days)) if date_err_days else float("nan"),
        "changepoint_date_err_days_p90": float(np.percentile(date_err_days, 90)) if date_err_days else float("nan"),
    }
    s = out["step"]
    print(f"  +15% step at midpoint: flagged {_pct(s['shift_flagged_pct'])}%, "
          f"changepoint located {_pct(s['changepoint_located_pct'])}%, "
          f"date error median {s['changepoint_date_err_days_median']:.0f} d "
          f"(p90 {s['changepoint_date_err_days_p90']:.0f} d)")
    return out


# --------------------------------------------------------------------------- #
# Study D — cross-validation on the bundled real dataset
# --------------------------------------------------------------------------- #
def study_d() -> dict:
    print("\n=== Study D: cross-validation on the bundled 10-year dataset ===")
    if not DEFAULT_DATASET.exists():
        print("  bundled dataset not found — skipping")
        return {"skipped": True}

    wf = RatingCurveWorkflow()
    val = wf.load_and_validate(DEFAULT_DATASET, sheet_name=DEFAULT_DATASET_SHEET)
    clean = wf.cleaned_df
    fitrows = clean[clean["is_valid"]].copy()
    n = len(fitrows)
    print(f"  {n} valid gaugings")

    out = {"n_valid_gaugings": int(n), "configs": []}
    for seg_label, seg in (("single", 1), ("auto-BIC", "auto")):
        loo = leave_one_out_error(fitrows, segments=seg)
        fit = fit_rating_curve(fitrows, segments=seg, n_bootstrap=1000,
                               ci_level=CI_LEVEL, random_state=0)
        rec = {
            "segments": seg_label,
            "n_segments_selected": int(fit.get("n_segments", 1)),
            "in_sample_r2": float(fit["r_squared"]),
            "loo_rmspe_pct": loo["rmspe_pct"] if loo else None,
            "loo_mae_pct": loo["mae_pct"] if loo else None,
            "loo_bias_pct": loo["bias_pct"] if loo else None,
            "loo_p95_abs_pct": loo["p95_abs_pct"] if loo else None,
        }
        bands = fit.get("bands") or {}
        rec["conf_band_halfwidth_pct_at_median"] = bands.get("ci_halfwidth_pct_at_median")
        rec["b_ci"] = list(bands["b_ci"]) if bands.get("b_ci") else None

        # temporal hold-out: fit on the earliest 70%, predict the latest 30%
        if DATE in fitrows.columns:
            ordered = fitrows.assign(_d=pd.to_datetime(fitrows[DATE], errors="coerce")) \
                             .dropna(subset=["_d"]).sort_values("_d")
            cut = int(len(ordered) * 0.7)
            train, test = ordered.iloc[:cut], ordered.iloc[cut:]
            try:
                f2 = fit_rating_curve(train, segments=seg, n_bootstrap=0,
                                      random_state=0, _diagnostics=False)
                pred = predict_discharge(f2, test[STAGE_M].to_numpy(float))
                obs = test[DISCHARGE_CMS].to_numpy(float)
                m = (obs > 0) & np.isfinite(pred)
                err = (pred[m] - obs[m]) / obs[m]
                rec["holdout_last30pct_rmspe_pct"] = float(100 * np.sqrt(np.mean(err ** 2)))
                rec["holdout_last30pct_n"] = int(m.sum())
            except Exception as exc:  # noqa: BLE001
                rec["holdout_error"] = str(exc)
        out["configs"].append(rec)
        print(f"  {seg_label}: {rec['n_segments_selected']} seg, "
              f"in-sample R² {rec['in_sample_r2']:.4f}, "
              f"LOO RMSPE {rec['loo_rmspe_pct']:.1f}%  MAE {rec['loo_mae_pct']:.1f}%  "
              f"bias {rec['loo_bias_pct']:+.1f}%; "
              f"hold-out last-30% RMSPE "
              f"{rec.get('holdout_last30pct_rmspe_pct', float('nan')):.1f}%")
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="fewer trials (~1 min)")
    ap.add_argument("--only", choices=["a", "b", "c", "d"], nargs="+",
                    help="run only these studies")
    args = ap.parse_args()
    run = set(args.only or ["a", "b", "c", "d"])

    t0 = time.time()
    results: dict = {
        "meta": {
            "package_version": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "generated": date.today().isoformat(),
            "ci_level": CI_LEVEL,
            "mode": "quick" if args.quick else "full",
        }
    }
    if "a" in run:
        results["study_a_single_power_law"] = study_a(args.quick)
    if "b" in run:
        results["study_b_piecewise_bic"] = study_b(args.quick)
    if "c" in run:
        results["study_c_drift_detection"] = study_c(args.quick)
    if "d" in run:
        results["study_d_real_data_cv"] = study_d()
    results["meta"]["runtime_seconds"] = round(time.time() - t0, 1)

    RESULTS_DIR.mkdir(exist_ok=True)
    dest = RESULTS_DIR / "simulation_study.json"
    dest.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {dest}  ({results['meta']['runtime_seconds']} s)")


if __name__ == "__main__":
    main()
