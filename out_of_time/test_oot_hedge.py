from pathlib import Path
import importlib.util, json
from math import exp
import numpy as np
import pandas as pd

PATH = Path(__file__).parent / "run_hedge.py"
spec = importlib.util.spec_from_file_location("oot_hedge_revalidation", PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_daily_rebalance_is_self_financing_with_calendar_funding():
    old_spot = np.array([100.0, 80.0])
    new_spot = np.array([103.0, 79.0])
    old = np.array([0.3, -0.2])
    new = np.array([0.31, -0.18])
    cash = -10.0
    r = 0.0431
    dt = 3 / 365
    c, wealth, funding, stock_pnl, trade_cash = m.cash_step(
        cash, old, new, old_spot, new_spot, r, dt
    )
    independently_held_wealth = cash * exp(r * dt) + float(old @ new_spot)
    assert abs(wealth - independently_held_wealth) < 1e-12
    assert abs(c + float(new @ new_spot) - independently_held_wealth) < 1e-12
    assert abs(stock_pnl - float(old @ (new_spot - old_spot))) < 1e-12


def test_time_to_expiry_and_weekend_reference_are_explicit():
    daily = pd.read_csv(
        PATH.parent / "results/daily_delta_hedge.csv", parse_dates=["date"]
    )
    observed = daily.loc[daily.is_market_observation]
    assert len(observed) == 251 and observed.date.max() == pd.Timestamp("2026-03-06")
    assert (observed.remaining_T > 0).all()
    np.testing.assert_allclose(
        observed.remaining_T, (m.EXPIRY - observed.date).dt.days / 365.0
    )
    assert daily.iloc[-1].date == pd.Timestamp("2026-03-07")
    assert not bool(daily.iloc[-1].is_market_observation)
    assert daily.iloc[-1].remaining_T == 0


def test_hedging_parameters_and_seed_remain_the_frozen_contract(inputs):
    report = json.loads((PATH.parent / "results/summary.json").read_text())
    assert report["model_inputs_frozen"] == inputs.__dict__
    assert (
        "500000" in report["primary_delta_method"]
        and "20260903" in report["primary_delta_method"]
    )
    assert report["payoff_is_single_realization_not_pricing_validation"] is True
