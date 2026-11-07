"""One-contract daily-close delta-hedging path; no parameter refits.

Primary deltas and model marks use the original 500,000-path CRN functions,
seed, 1% spot bumps and fixed h. Exact-integral deltas are used only to explain
numerical hedging error, not to replace the primary hedge.
"""

from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from math import exp
import hashlib, json, sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from run_analysis import load_inputs, PATHS, SEED
from spread_option_pricer import (
    bump_and_revalue,
    price_spread_option,
    exact_conditional_price,
    exact_finite_difference,
    estimate_parameters,
)

EXPIRY = pd.Timestamp("2026-03-07")


def local_gammas(inputs):
    """FD of the exact-model price, same 1e-4 relative spot benchmark step."""
    h1, h2 = inputs.spot_1 * 1e-4, inputs.spot_2 * 1e-4
    value = exact_conditional_price(inputs)

    def v(a, b):
        return exact_conditional_price(
            replace(inputs, spot_1=inputs.spot_1 + a, spot_2=inputs.spot_2 + b)
        )

    return (
        (v(h1, 0) - 2 * value + v(-h1, 0)) / (h1 * h1),
        (v(0, h2) - 2 * value + v(0, -h2)) / (h2 * h2),
        (v(h1, h2) - v(h1, -h2) - v(-h1, h2) + v(-h1, -h2)) / (4 * h1 * h2),
    )


def cash_step(cash, old_delta, new_delta, old_spot, new_spot, rate, dt):
    funding = cash * (exp(rate * dt) - 1)
    stock_pnl = float(np.dot(old_delta, new_spot - old_spot))
    before = cash + funding + float(np.dot(old_delta, new_spot))
    trade_cash = -float(np.dot(new_delta - old_delta, new_spot))
    after_cash = cash + funding + trade_cash
    assert np.isclose(
        before, after_cash + float(np.dot(new_delta, new_spot)), atol=1e-10
    )
    return after_cash, before, funding, stock_pnl, trade_cash


def main():
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    path = ROOT / "data/tsla_tsm_close.csv"
    expected = manifest["files"]["tsla_tsm_close.csv"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Price data hash mismatch")
    prices = pd.read_csv(path, index_col="date", parse_dates=True).sort_index()
    prices = prices.loc[
        (prices.index >= pd.Timestamp("2025-03-07")) & (prices.index <= EXPIRY)
    ]
    if prices.index[0] != pd.Timestamp("2025-03-07") or prices.isna().any().any():
        raise ValueError("Incomplete paired close history")
    frozen = load_inputs()[1]
    premium = price_spread_option(frozen, PATHS, SEED).price
    rows = []
    gammas = []
    for n, (date, row) in enumerate(prices.iterrows()):
        candidate = replace(
            frozen,
            spot_1=float(row.TSLA),
            spot_2=float(row.TSM),
            maturity=(EXPIRY - date).days / 365.0,
        )
        mark = price_spread_option(candidate, PATHS, SEED).price
        delta = np.array(
            [
                bump_and_revalue(
                    candidate, field, getattr(candidate, field) * 0.01, PATHS, SEED
                )
                for field in ["spot_1", "spot_2"]
            ]
        )
        exact_delta = np.array(
            [
                exact_finite_difference(
                    candidate, field, getattr(candidate, field) * 1e-4
                )
                for field in ["spot_1", "spot_2"]
            ]
        )
        rows.append(
            {
                "date": date,
                "TSLA": float(row.TSLA),
                "TSM": float(row.TSM),
                "remaining_T": candidate.maturity,
                "model_mc_mark": mark,
                "exact_model_mark": exact_conditional_price(candidate),
                "delta_tsla": delta[0],
                "delta_tsm": delta[1],
                "exact_delta_tsla": exact_delta[0],
                "exact_delta_tsm": exact_delta[1],
                "is_market_observation": True,
            }
        )
        gammas.append(local_gammas(candidate))
        if (n + 1) % 50 == 0:
            print(
                f"Frozen-model marks and deltas: {n+1}/{len(prices)} closes", flush=True
            )
    # Calendar expiry is a Saturday. The supplied contract has no holiday rule.
    # Use the last PRECEDING close as an explicitly labelled settlement proxy;
    # keep a one-calendar-day cash-funding stub without inventing a market print.
    last = rows[-1]
    payoff = max(last["TSLA"] - frozen.hedge_ratio * last["TSM"] - frozen.strike, 0.0)
    settlement = {
        **last,
        "date": EXPIRY,
        "remaining_T": 0.0,
        "model_mc_mark": payoff,
        "exact_model_mark": payoff,
        "delta_tsla": 0.0,
        "delta_tsm": 0.0,
        "exact_delta_tsla": 0.0,
        "exact_delta_tsm": 0.0,
        "is_market_observation": False,
    }
    rows.append(settlement)
    cash = shadow_cash = None
    previous_error = 0.0
    ledger = []
    for i, row in enumerate(rows):
        spot = np.array([row["TSLA"], row["TSM"]])
        delta = np.array([row["delta_tsla"], row["delta_tsm"]])
        shadow_delta = np.array([row["exact_delta_tsla"], row["exact_delta_tsm"]])
        if i == 0:
            cash = premium - float(delta @ spot)
            shadow_cash = premium - float(shadow_delta @ spot)
            wealth = shadow_wealth = premium
            funding = stock_pnl = trade_cash = 0.0
            # Initial contract is issued at the main premium.
            row["model_mc_mark"] = premium
            dt = 0.0
            vol1 = vol2 = covariance = 0.0
        else:
            prev = rows[i - 1]
            old_spot = np.array([prev["TSLA"], prev["TSM"]])
            old_delta = np.array([prev["delta_tsla"], prev["delta_tsm"]])
            old_shadow = np.array([prev["exact_delta_tsla"], prev["exact_delta_tsm"]])
            dt = (row["date"] - prev["date"]).days / 365.0
            cash, wealth, funding, stock_pnl, trade_cash = cash_step(
                cash, old_delta, delta, old_spot, spot, frozen.risk_free_rate, dt
            )
            shadow_cash, shadow_wealth, *_ = cash_step(
                shadow_cash,
                old_shadow,
                shadow_delta,
                old_spot,
                spot,
                frozen.risk_free_rate,
                dt,
            )
            dS = spot - old_spot
            g11, g22, g12 = gammas[i - 1]
            vol1 = (
                -0.5
                * g11
                * (dS[0] ** 2 - frozen.volatility_1**2 * old_spot[0] ** 2 * dt)
            )
            vol2 = (
                -0.5
                * g22
                * (dS[1] ** 2 - frozen.volatility_2**2 * old_spot[1] ** 2 * dt)
            )
            covariance = -g12 * (
                dS[0] * dS[1]
                - frozen.correlation
                * frozen.volatility_1
                * frozen.volatility_2
                * old_spot[0]
                * old_spot[1]
                * dt
            )
        error = wealth - row["model_mc_mark"]
        grow = exp(frozen.risk_free_rate * (EXPIRY - row["date"]).days / 365.0)
        ledger.append(
            {
                **row,
                "dt_years": dt,
                "stock_hedge_pnl": stock_pnl,
                "cash_funding_pnl": funding,
                "rebalancing_cashflow": trade_cash,
                "cash_after_rebalance": cash,
                "hedge_assets_value": wealth,
                "cumulative_short_option_hedge_pnl": error,
                "daily_short_option_hedge_pnl": error - previous_error,
                "exact_delta_attribution_assets_value": shadow_wealth,
                "gamma_qv_tsla_to_expiry": vol1 * grow,
                "gamma_qv_tsm_to_expiry": vol2 * grow,
                "gamma_covariance_to_expiry": covariance * grow,
            }
        )
        previous_error = error
    daily = pd.DataFrame(ledger)
    terminal = float(daily.iloc[-1].cumulative_short_option_hedge_pnl)
    initial_sampling = (premium - daily.iloc[0].exact_model_mark) * exp(
        frozen.risk_free_rate * frozen.maturity
    )
    numerical_delta = float(
        daily.iloc[-1].hedge_assets_value
        - daily.iloc[-1].exact_delta_attribution_assets_value
    )
    vol_tsla = float(daily.gamma_qv_tsla_to_expiry.sum())
    vol_tsm = float(daily.gamma_qv_tsm_to_expiry.sum())
    covariance = float(daily.gamma_covariance_to_expiry.sum())
    residual = (
        terminal - initial_sampling - numerical_delta - vol_tsla - vol_tsm - covariance
    )
    attribution = [
        {"component": k, "terminal_dollars": v}
        for k, v in [
            ("initial_main_MC_price_minus_exact_model", initial_sampling),
            ("main_CRN_delta_vs_exact_FD_delta", numerical_delta),
            ("TSLA_realized_quadratic_variation_vs_model", vol_tsla),
            ("TSM_realized_quadratic_variation_vs_model", vol_tsm),
            ("realized_cross_variation_vs_model_covariance", covariance),
            ("discrete_hedging_higher_order_and_carry_residual", residual),
        ]
    ]
    realized = estimate_parameters(prices)
    observed = daily.loc[daily.is_market_observation].iloc[1:]
    summary = {
        "status": "COMPLETED_SINGLE_PATH",
        "valuation_date": "2025-03-07",
        "calendar_expiry": "2026-03-07",
        "settlement_reference_date": str(prices.index[-1].date()),
        "settlement_convention": "Preceding trading close proxy; original contract does not specify a holiday adjustment. No Saturday market close exists.",
        "settlement_closes": {
            "TSLA": float(prices.iloc[-1].TSLA),
            "TSM": float(prices.iloc[-1].TSM),
        },
        "payoff_at_reference_close": payoff,
        "payoff_is_single_realization_not_pricing_validation": True,
        "primary_delta_method": "Original CRN central 1%-spot bump, 500000 paths, seed 20260903; fixed h and sigma/rho/r.",
        "premium_received": premium,
        "model_inputs_frozen": frozen.__dict__,
        "actual_market_closes": len(prices),
        "observed_return_intervals": len(observed),
        "terminal_short_option_delta_hedged_pnl": terminal,
        "daily_hedging_error_std_dollars_ddof1": float(
            observed.daily_short_option_hedge_pnl.std(ddof=1)
        ),
        "daily_hedging_error_mean_dollars": float(
            observed.daily_short_option_hedge_pnl.mean()
        ),
        "realized_simple_return_estimates": realized,
        "model_vs_realized": {
            "sigma_tsla_model": frozen.volatility_1,
            "sigma_tsla_realized": realized["volatility_1"],
            "sigma_tsm_model": frozen.volatility_2,
            "sigma_tsm_realized": realized["volatility_2"],
            "rho_model": frozen.correlation,
            "rho_realized": realized["correlation"],
        },
        "error_attribution": attribution,
        "data_sha256": expected,
        "frictions": "Frictionless zero-dividend model ledger: zero transaction fees, zero stock borrow fees, zero cash dividends. Cash earns/pays the original r on ACT/365 calendar intervals.",
        "interpretation": "A single realized path cannot establish pricing accuracy or terminal hedge-error variance. Gamma-weighted variation attribution is a second-order accounting approximation, not a causal identification of the error. Residual includes discrete hedge/jump/higher-order effects; omitted real dividends/fees are an additional limitation.",
    }
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    # LSEG close prices stay in memory for P&L accounting; the published
    # ledger drops them so licensed prices are not redistributed.
    daily.drop(columns=["TSLA", "TSM"]).to_csv(out / "daily_delta_hedge.csv", index=False)
    pd.DataFrame(attribution).to_csv(out / "hedge_error_attribution.csv", index=False)
    pd.DataFrame([summary["model_vs_realized"]]).to_csv(
        out / "model_vs_realized.csv", index=False
    )
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(
        daily.date,
        daily.cumulative_short_option_hedge_pnl,
        label="Cumulative short-call delta-hedged P&L",
    )
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set_ylabel("Dollars per payoff unit")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    axes[1].bar(daily.date, daily.daily_short_option_hedge_pnl, width=1.2)
    axes[1].set_ylabel("Daily hedging P&L ($)")
    axes[1].grid(alpha=0.2)
    fig.suptitle("Frozen TSLA/TSM model: one observed daily-close hedging path")
    fig.tight_layout()
    fig.savefig(out / "delta_hedge_pnl.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
