"""Reproduce the TSLA/TSM main results and the validation suite."""

from __future__ import annotations
import json
import hashlib
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spread_option_pricer import (
    MarketInputs,
    adaptive_conditional_price,
    bump_and_revalue,
    correlated_normals,
    discounted_payoffs,
    estimate_parameters,
    exact_conditional_price,
    exact_finite_difference,
    kirk_approximation,
    pathwise_greeks,
    price_spread_option,
    summarize_payoffs,
    variance_reduced_price,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "tsla_tsm_prices_2020_2025.csv"
RESULTS_DIR = ROOT / "results"
SEED, PATHS = 20260903, 500_000
VALUATION_DATE = "2025-03-07"


def load_inputs() -> tuple[pd.DataFrame | None, MarketInputs, dict]:
    """Calibrated contract inputs.

    The price panel is a Bloomberg export and is not redistributed. When the
    file is absent, fall back to the same calibrated inputs recorded in
    results/summary.json so the pricing, quadrature and Greek work still runs;
    panel-dependent steps (calibration checks, assumption comparisons) are
    skipped by the callers.
    """
    if not DATA_PATH.is_file():
        recorded = json.loads((RESULTS_DIR / "summary.json").read_text())
        print(
            f"Price panel {DATA_PATH.name} not found (Bloomberg export, not "
            "redistributed). Using the calibrated inputs recorded in "
            "results/summary.json; calibration and assumption comparisons are "
            "skipped. See data/README.md to restore the panel."
        )
        return None, MarketInputs(**recorded["model_inputs"]), recorded["estimated_parameters"]
    prices = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date")
    calibration = prices.loc[prices["date"].between("2024-01-01", "2024-12-31")]
    estimates = estimate_parameters(calibration)
    valuation_row = prices.loc[prices["date"] <= VALUATION_DATE].iloc[-1]
    s1, s2 = float(valuation_row["TSLA"]), float(valuation_row["TSM"])
    inputs = MarketInputs(
        s1,
        s2,
        estimates["volatility_1"],
        estimates["volatility_2"],
        estimates["correlation"],
        0.0431,
        1.0,
        200.0,
        s1 / s2,
    )
    return prices, inputs, estimates


def convergence_analysis(inputs: MarketInputs) -> pd.DataFrame:
    counts = [2_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000, PATHS]
    z1, z2 = correlated_normals(PATHS, inputs.correlation, SEED)
    payoffs = discounted_payoffs(inputs, z1, z2)
    return pd.DataFrame(
        [asdict(summarize_payoffs(payoffs[:n], n, SEED)) for n in counts]
    )


def variance_comparison(
    inputs: MarketInputs, exact: float
) -> tuple[pd.DataFrame, dict]:
    start = perf_counter()
    baseline = price_spread_option(inputs, PATHS, SEED)
    elapsed = perf_counter() - start
    rows = [
        {
            "method": "plain_mc",
            **asdict(baseline),
            "seconds": elapsed,
            "payoff_evaluations": PATHS,
            "independent_samples": PATHS,
        }
    ]
    details = {}
    for method in ("antithetic", "control_variate", "conditional_mc"):
        start = perf_counter()
        result, details[method] = variance_reduced_price(inputs, method, PATHS, SEED)
        elapsed = perf_counter() - start
        rows.append(
            {
                "method": method,
                **asdict(result),
                "seconds": elapsed,
                "payoff_evaluations": details[method]["payoff_evaluations"],
                "independent_samples": details[method]["independent_samples"],
            }
        )
    frame = pd.DataFrame(rows)
    frame["error_vs_exact"] = frame["price"] - exact
    frame["relative_error_vs_exact"] = frame["error_vs_exact"] / exact
    frame["variance_reduction_factor"] = (
        baseline.standard_error / frame["standard_error"]
    ) ** 2
    return frame, details


def greek_comparison(inputs: MarketInputs) -> tuple[pd.DataFrame, dict]:
    specs = [
        ("delta_tsla", "spot_1", inputs.spot_1 * 0.01, inputs.spot_1 * 1e-4, 1.0),
        ("delta_tsm", "spot_2", inputs.spot_2 * 0.01, inputs.spot_2 * 1e-4, 1.0),
        ("vega_tsla_per_1pct", "volatility_1", 0.01, 1e-4, 0.01),
        ("vega_tsm_per_1pct", "volatility_2", 0.01, 1e-4, 0.01),
        ("correlation_sensitivity_per_1pct", "correlation", 0.01, 1e-4, 0.01),
    ]
    pw = pathwise_greeks(inputs, PATHS, SEED)
    rows, original = [], {}
    for name, field, bump, exact_bump, scale in specs:
        original[name] = bump_and_revalue(inputs, field, bump, PATHS, SEED) * scale
        rows.append(
            {
                "greek": name,
                "unit": "per dollar" if scale == 1 else "per +0.01 absolute parameter",
                "crn_bump": original[name],
                "crn_bump_size": bump,
                "pathwise": pw[field]["estimate"] * scale if field in pw else None,
                "pathwise_standard_error": (
                    pw[field]["standard_error"] * scale if field in pw else None
                ),
                "exact_finite_difference": exact_finite_difference(
                    inputs, field, exact_bump
                )
                * scale,
                "exact_bump_size": exact_bump,
            }
        )
    return pd.DataFrame(rows), original


def assumption_comparison(
    prices: pd.DataFrame, inputs: MarketInputs, exact: float
) -> pd.DataFrame:
    ancillary = json.loads((ROOT / "data" / "assumption_inputs.json").read_text())
    calendar = prices.loc[prices["date"].between("2024-01-01", "2024-12-31")]
    # Strictly before the valuation date: 252 closes -> 251 returns.
    recent = prices.loc[prices["date"] < VALUATION_DATE].tail(252)

    def recalibrate(sample, return_type="simple"):
        est = estimate_parameters(sample, return_type=return_type)
        return replace(
            inputs,
            **{k: est[k] for k in ["volatility_1", "volatility_2", "correlation"]},
        )

    payments = ancillary["tsm_dividend"]["payments"]
    q2 = sum(p["usd_per_ads"] for p in payments) / inputs.spot_2
    cases = [
        ("main_simple_2024", inputs, calendar, "simple"),
        ("log_returns_2024", recalibrate(calendar, "log"), calendar, "log"),
        ("immediately_prior_252_closes", recalibrate(recent), recent, "simple"),
        (
            "tsm_trailing_cash_dividend_yield",
            replace(inputs, dividend_yield_2=q2),
            calendar,
            "simple",
        ),
        (
            "valuation_date_1y_treasury_rate",
            replace(
                inputs, risk_free_rate=ancillary["treasury_1y"]["continuous_proxy"]
            ),
            calendar,
            "simple",
        ),
    ]
    baseline_mc = price_spread_option(inputs, PATHS, SEED).price
    rows = []
    for name, candidate, sample, returns in cases:
        price = exact_conditional_price(candidate)
        mc = price_spread_option(candidate, PATHS, SEED).price
        rows.append(
            {
                "assumption": name,
                "exact_price": price,
                "change_vs_main_exact": price - exact,
                "mc_price_crn": mc,
                "change_vs_main_mc_crn": mc - baseline_mc,
                "return_type": returns,
                "calibration_start": str(sample["date"].min().date()),
                "calibration_end": str(sample["date"].max().date()),
                "price_observations": len(sample),
                "return_observations": len(sample) - 1,
                "volatility_1": candidate.volatility_1,
                "volatility_2": candidate.volatility_2,
                "correlation": candidate.correlation,
                "risk_free_rate": candidate.risk_free_rate,
                "dividend_yield_2": candidate.dividend_yield_2,
            }
        )
    return pd.DataFrame(rows)


def save_plots(convergence, variance, greeks, assumptions, exact, kirk):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(
        convergence["paths"],
        convergence["price"],
        yerr=1.96 * convergence["standard_error"],
        fmt="o-",
        capsize=3,
        label="MC and 95% confidence interval",
    )
    ax.axhline(exact, label=f"Conditional integral: {exact:.4f}", color="black")
    ax.axhline(kirk, label=f"Kirk: {kirk:.4f}", color="tab:red", linestyle="--")
    ax.set(
        xscale="log",
        xlabel="Paths (nested prefixes of the main draws)",
        ylabel="Price ($)",
        title="Monte Carlo convergence",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "convergence.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].errorbar(
        variance["method"],
        variance["price"],
        yerr=1.96 * variance["standard_error"],
        fmt="o",
        capsize=4,
    )
    axes[0].axhline(exact, color="black", label="Integral benchmark")
    axes[0].set(
        ylabel="Price and 95% CI ($)", title="Separate variance-reduction estimators"
    )
    axes[1].bar(variance["method"], variance["variance_reduction_factor"])
    axes[1].set(
        yscale="log", ylabel="SE(plain MC)^2 / SE(method)^2", title="Variance reduction"
    )
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "variance_reduction.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, names, label in [
        (axes[0], ["delta_tsla", "delta_tsm"], "Delta per $1 spot"),
        (
            axes[1],
            ["vega_tsla_per_1pct", "vega_tsm_per_1pct"],
            "Vega per +1 percentage point",
        ),
    ]:
        part = greeks.set_index("greek").loc[names]
        for j, (field, legend) in enumerate(
            [
                ("crn_bump", "CRN bump"),
                ("pathwise", "Pathwise"),
                ("exact_finite_difference", "Integral FD"),
            ]
        ):
            ax.scatter(np.arange(2) + (j - 1) * 0.08, part[field], label=legend)
        ax.set(xticks=[0, 1], xticklabels=["TSLA", "TSM"], ylabel=label)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Greeks with the contract hedge ratio held fixed")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "greeks.png", dpi=160)
    plt.close(fig)
    if assumptions is None:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    changes = assumptions.iloc[1:]
    ax.barh(
        [
            "Log returns",
            "Prior 252 closes",
            "TSM dividend yield",
            "Valuation-date 1y rate",
        ],
        changes["change_vs_main_exact"],
    )
    ax.axvline(0, color="black")
    ax.set(
        xlabel="Price change vs main integral ($)",
        title="One-factor assumption comparisons",
    )
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "assumptions.png", dpi=160)
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    prices, inputs, estimates = load_inputs()
    result = price_spread_option(inputs, PATHS, SEED)
    kirk = kirk_approximation(inputs)
    adaptive, adaptive_error = adaptive_conditional_price(inputs)
    quadrature = pd.DataFrame(
        [
            {
                "method": "gauss_hermite",
                "nodes": n,
                "price": exact_conditional_price(inputs, n),
                "error_estimate": None,
            }
            for n in (100, 128, 256)
        ]
        + [
            {
                "method": "adaptive_quad",
                "nodes": None,
                "price": adaptive,
                "error_estimate": adaptive_error,
            }
        ]
    )
    exact = exact_conditional_price(inputs, 128)
    variance, vr_details = variance_comparison(inputs, exact)
    greeks, risk = greek_comparison(inputs)
    convergence = convergence_analysis(inputs)
    assumptions = (
        assumption_comparison(prices, inputs, exact) if prices is not None else None
    )
    if prices is not None:
        source_period = {
            "price_history_start": str(prices["date"].min().date()),
            "price_history_end": str(prices["date"].max().date()),
            "calibration_window": "2024-01-02 to 2024-12-31",
            "valuation_date": VALUATION_DATE,
            "calibration_prices": 252,
            "calibration_returns": 251,
            "return_type": "simple",
        }
        data_sha256 = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    else:
        recorded = json.loads((RESULTS_DIR / "summary.json").read_text())
        source_period = recorded["source_period"]
        data_sha256 = recorded["data_sha256"]
    summary = {
        "source_period": source_period,
        "data_sha256": data_sha256,
        "estimated_parameters": estimates,
        "model_inputs": asdict(inputs),
        "monte_carlo": asdict(result),
        "kirk_approximation": kirk,
        "monte_carlo_minus_kirk": result.price - kirk,
        "relative_difference": (result.price - kirk) / kirk,
        "exact_conditional_integral": exact,
        "quadrature_nodes": 128,
        "adaptive_quadrature_check": {
            "price": adaptive,
            "estimated_error": adaptive_error,
            "difference_vs_gh128": adaptive - exact,
        },
        "quadrature_128_vs_256_difference": abs(
            exact - exact_conditional_price(inputs, 256)
        ),
        "kirk_relative_error_vs_exact": (kirk - exact) / exact,
        "mc_error_in_standard_errors": (result.price - exact) / result.standard_error,
        "bump_and_revalue_risk": risk,
        "variance_reduction_details": vr_details,
        "greeks_contract_convention": "h fixed for all spot bumps and pathwise derivatives",
        "limitations": [
            "Historical calibration; constant risk-neutral GBM volatility and correlation.",
            "No dividends and the original 4.31% rate in the main result.",
            "Dividend and historical-rate inputs are isolated one-factor proxy comparisons.",
            "Exact means numerical evaluation of this GBM model, not an observed market price.",
        ],
    }
    frames = [
        ("quadrature", quadrature),
        ("convergence", convergence),
        ("variance_reduction", variance),
        ("greeks", greeks),
    ]
    if assumptions is not None:
        frames.append(("assumption_sensitivity", assumptions))
    for name, frame in frames:
        frame.to_csv(RESULTS_DIR / f"{name}.csv", index=False)
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    save_plots(convergence, variance, greeks, assumptions, exact, kirk)
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(
        "\nVARIANCE REDUCTION\n"
        + variance[
            [
                "method",
                "price",
                "standard_error",
                "error_vs_exact",
                "variance_reduction_factor",
                "seconds",
            ]
        ].to_string(index=False)
    )
    print("\nGREEKS\n" + greeks.to_string(index=False))
    if assumptions is not None:
        print(
            "\nASSUMPTIONS\n"
            + assumptions[
                [
                    "assumption",
                    "exact_price",
                    "change_vs_main_exact",
                    "mc_price_crn",
                    "change_vs_main_mc_crn",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
