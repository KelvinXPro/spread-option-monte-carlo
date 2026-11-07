"""Lock the main numbers and check independent model identities."""

from dataclasses import replace
from math import exp

import numpy as np
import pytest

from run_analysis import DATA_PATH, PATHS, SEED, assumption_comparison, load_inputs
from spread_option_pricer import (
    correlated_normals,
    exact_conditional_price,
    estimate_parameters,
    kirk_approximation,
    margrabe_price,
    pathwise_greeks,
    price_spread_option,
    bump_and_revalue,
    variance_reduced_price,
)


@pytest.fixture(scope="module")
def main_result(inputs):
    return price_spread_option(inputs, PATHS, SEED)


def test_main_calibration_is_252_prices_251_simple_returns(price_panel):
    prices, inputs, est = load_inputs()
    calibration = prices.loc[prices["date"].between("2024-01-01", "2024-12-31")]
    assert len(calibration) == 252 and est["observations"] == 251
    ret = calibration[["TSLA", "TSM"]].pct_change(fill_method=None).dropna()
    assert inputs.volatility_1 == pytest.approx(ret["TSLA"].std(ddof=1) * np.sqrt(252))
    assert inputs.correlation == pytest.approx(ret.corr().iloc[0, 1])
    assert (
        inputs.risk_free_rate == 0.0431
        and inputs.dividend_yield_1 == inputs.dividend_yield_2 == 0
    )
    assert inputs.maturity == 1 and inputs.strike == 200
    assert inputs.hedge_ratio == inputs.spot_1 / inputs.spot_2


def test_frozen_main_numbers_are_locked(inputs, main_result):
    assert PATHS == 500_000 and SEED == 20260903
    assert main_result.price == pytest.approx(21.561243548159304, abs=1e-10)
    assert main_result.standard_error == pytest.approx(0.12912829126483363, abs=1e-12)
    kirk = kirk_approximation(inputs)
    assert kirk == pytest.approx(21.79599447333502, abs=1e-10)
    assert (main_result.price - kirk) / kirk == pytest.approx(
        -0.010770370008255831, abs=1e-12
    )


def test_quadrature_independently_converges(inputs):
    values = [exact_conditional_price(inputs, n) for n in (100, 128, 256)]
    assert max(values) - min(values) < 1e-10
    assert values[1] == pytest.approx(21.8148437640918, abs=1e-10)


def test_kirk_and_mc_are_each_within_three_main_standard_errors_of_exact(
    inputs, main_result
):
    exact = exact_conditional_price(inputs)
    assert abs(kirk_approximation(inputs) - exact) < 3 * main_result.standard_error
    assert abs(main_result.price - exact) < 3 * main_result.standard_error
    assert (main_result.price - exact) / main_result.standard_error == pytest.approx(
        -1.964, abs=0.002
    )


@pytest.mark.parametrize("q1,q2", [(0.0, 0.0), (0.01, 0.025)])
def test_zero_strike_matches_margrabe_identity(inputs, q1, q2):
    zero = replace(inputs, strike=0, dividend_yield_1=q1, dividend_yield_2=q2)
    identity = margrabe_price(zero)
    assert exact_conditional_price(zero) == pytest.approx(identity, abs=1e-10)
    assert kirk_approximation(zero) == pytest.approx(identity, abs=1e-10)


@pytest.mark.parametrize("method", ["antithetic", "control_variate", "conditional_mc"])
def test_each_variance_reduction_method_lowers_standard_error(
    inputs, main_result, method
):
    result, details = variance_reduced_price(inputs, method, PATHS, SEED)
    assert result.standard_error < main_result.standard_error
    assert (
        abs(result.price - exact_conditional_price(inputs)) < 3 * result.standard_error
    )
    if method == "antithetic":
        assert details["independent_samples"] == PATHS // 2
    if method == "control_variate":
        assert details["pilot_paths"] == 10_000
        np.testing.assert_allclose(
            details["analytic_control_expectations"],
            [inputs.spot_1, inputs.hedge_ratio * inputs.spot_2],
        )


@pytest.mark.parametrize("field", ["spot_1", "spot_2"])
def test_pathwise_delta_matches_crn_bump(inputs, field):
    pw = pathwise_greeks(inputs)[field]["estimate"]
    bump = bump_and_revalue(inputs, field, getattr(inputs, field) * 0.01)
    assert pw == pytest.approx(bump, abs=2e-4)


@pytest.mark.parametrize("field", ["volatility_1", "volatility_2"])
def test_pathwise_vega_matches_crn_bump(inputs, field):
    pw = pathwise_greeks(inputs)[field]["estimate"]
    bump = bump_and_revalue(inputs, field, 0.01)
    assert pw == pytest.approx(bump, abs=0.03)


def test_generated_normal_correlation_matches_target(inputs):
    z1, z2 = correlated_normals(250_000, inputs.correlation, 7)
    assert np.corrcoef(z1, z2)[0, 1] == pytest.approx(inputs.correlation, abs=0.01)


def test_call_monotonicity_under_common_random_numbers(inputs):
    low = price_spread_option(replace(inputs, spot_1=240), 100_000, 29).price
    high = price_spread_option(replace(inputs, spot_1=280), 100_000, 29).price
    assert high > low
    assert exact_conditional_price(
        replace(inputs, strike=150)
    ) > exact_conditional_price(replace(inputs, strike=250))


def test_zero_volatility_matches_deterministic_payoff(inputs):
    deterministic = replace(inputs, volatility_1=0, volatility_2=0, spot_1=600.0)
    expected = exp(-inputs.risk_free_rate) * max(
        600 * exp(inputs.risk_free_rate)
        - inputs.hedge_ratio * inputs.spot_2 * exp(inputs.risk_free_rate)
        - inputs.strike,
        0,
    )
    assert price_spread_option(deterministic, 10_000, 37).price == pytest.approx(
        expected, abs=1e-10
    )
    assert exact_conditional_price(deterministic) == pytest.approx(expected, abs=1e-10)


def test_assumption_comparisons_are_separate_and_exclude_valuation_day(
    inputs, price_panel
):
    prices, _, _ = load_inputs()
    frame = assumption_comparison(
        prices, inputs, exact_conditional_price(inputs)
    ).set_index("assumption")
    prior = frame.loc["immediately_prior_252_closes"]
    assert prior["price_observations"] == 252 and prior["return_observations"] == 251
    assert prior["calibration_end"] == "2025-03-06"
    assert frame.loc["main_simple_2024", "risk_free_rate"] == 0.0431
    assert frame.loc["main_simple_2024", "dividend_yield_2"] == 0
    assert frame.loc["tsm_trailing_cash_dividend_yield", "dividend_yield_2"] > 0
    assert (frame["return_observations"] == 251).all()


def test_gauss_hermite_agrees_with_adaptive_quadrature(inputs):
    from spread_option_pricer import adaptive_conditional_price

    price, error = adaptive_conditional_price(inputs)
    assert abs(price - exact_conditional_price(inputs)) < 1e-10
    assert error < 1e-8
