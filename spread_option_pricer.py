"""Monte Carlo pricing utilities for a two-asset equity spread option.

The option payoff is max(S1_T - hedge_ratio * S2_T - strike, 0).
Both assets follow correlated geometric Brownian motions under the
risk-neutral measure. Assumptions and random-number generation are kept
explicit so every number can be checked and reproduced.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import erf, exp, log, pi, sqrt

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import ndtr, roots_hermitenorm


@dataclass(frozen=True)
class MarketInputs:
    spot_1: float
    spot_2: float
    volatility_1: float
    volatility_2: float
    correlation: float
    risk_free_rate: float
    maturity: float
    strike: float
    hedge_ratio: float
    dividend_yield_1: float = 0.0
    dividend_yield_2: float = 0.0

    def validate(self) -> None:
        if self.spot_1 <= 0 or self.spot_2 <= 0:
            raise ValueError("Spot prices must be positive.")
        if self.volatility_1 < 0 or self.volatility_2 < 0:
            raise ValueError("Volatilities cannot be negative.")
        if not -1.0 <= self.correlation <= 1.0:
            raise ValueError("Correlation must be between -1 and 1.")
        if self.maturity <= 0:
            raise ValueError("Maturity must be positive.")
        if self.strike < 0 or self.hedge_ratio <= 0:
            raise ValueError("Strike must be nonnegative and hedge ratio positive.")


@dataclass(frozen=True)
class MonteCarloResult:
    price: float
    standard_error: float
    confidence_low: float
    confidence_high: float
    paths: int
    seed: int


def estimate_parameters(
    prices: pd.DataFrame, trading_days: int = 252, return_type: str = "simple"
) -> dict[str, float]:
    """Estimate annualized volatility and correlation from daily simple returns."""
    required = {"TSLA", "TSM"}
    if not required.issubset(prices.columns):
        raise ValueError(
            f"Missing required columns: {sorted(required - set(prices.columns))}"
        )
    if return_type == "simple":
        returns = prices[["TSLA", "TSM"]].pct_change(fill_method=None).dropna()
    elif return_type == "log":
        returns = np.log(prices[["TSLA", "TSM"]]).diff().dropna()
    else:
        raise ValueError("return_type must be simple or log")
    if len(returns) < 2:
        raise ValueError("At least three price observations are required.")
    return {
        "volatility_1": float(returns["TSLA"].std(ddof=1) * sqrt(trading_days)),
        "volatility_2": float(returns["TSM"].std(ddof=1) * sqrt(trading_days)),
        "correlation": float(returns["TSLA"].corr(returns["TSM"])),
        "annualized_mean_return_1": float(returns["TSLA"].mean() * trading_days),
        "annualized_mean_return_2": float(returns["TSM"].mean() * trading_days),
        "observations": int(len(returns)),
    }


def correlated_normals(
    paths: int, correlation: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if paths < 2:
        raise ValueError("At least two Monte Carlo paths are required.")
    if not -1.0 <= correlation <= 1.0:
        raise ValueError("Correlation must be between -1 and 1.")
    rng = np.random.default_rng(seed)
    z_1 = rng.standard_normal(paths)
    independent = rng.standard_normal(paths)
    z_2 = correlation * z_1 + sqrt(max(0.0, 1.0 - correlation**2)) * independent
    return z_1, z_2


def terminal_prices(
    inputs: MarketInputs, z_1: np.ndarray, z_2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    inputs.validate()
    if z_1.shape != z_2.shape:
        raise ValueError("The two random-normal arrays must have the same shape.")
    root_t = sqrt(inputs.maturity)
    terminal_1 = inputs.spot_1 * np.exp(
        (inputs.risk_free_rate - inputs.dividend_yield_1 - 0.5 * inputs.volatility_1**2)
        * inputs.maturity
        + inputs.volatility_1 * root_t * z_1
    )
    terminal_2 = inputs.spot_2 * np.exp(
        (inputs.risk_free_rate - inputs.dividend_yield_2 - 0.5 * inputs.volatility_2**2)
        * inputs.maturity
        + inputs.volatility_2 * root_t * z_2
    )
    return terminal_1, terminal_2


def discounted_payoffs(
    inputs: MarketInputs, z_1: np.ndarray, z_2: np.ndarray
) -> np.ndarray:
    terminal_1, terminal_2 = terminal_prices(inputs, z_1, z_2)
    payoff = np.maximum(
        terminal_1 - inputs.hedge_ratio * terminal_2 - inputs.strike, 0.0
    )
    return exp(-inputs.risk_free_rate * inputs.maturity) * payoff


def summarize_payoffs(
    discounted: np.ndarray, paths: int, seed: int
) -> MonteCarloResult:
    price = float(discounted.mean())
    standard_error = float(discounted.std(ddof=1) / sqrt(paths))
    half_width = 1.96 * standard_error
    return MonteCarloResult(
        price=price,
        standard_error=standard_error,
        confidence_low=price - half_width,
        confidence_high=price + half_width,
        paths=paths,
        seed=seed,
    )


def price_spread_option(
    inputs: MarketInputs, paths: int = 500_000, seed: int = 20260903
) -> MonteCarloResult:
    z_1, z_2 = correlated_normals(paths, inputs.correlation, seed)
    return summarize_payoffs(discounted_payoffs(inputs, z_1, z_2), paths, seed)


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def kirk_approximation(inputs: MarketInputs) -> float:
    """Kirk approximation for max(S1 - h*S2 - K, 0)."""
    inputs.validate()
    forward_1 = inputs.spot_1 * exp(
        (inputs.risk_free_rate - inputs.dividend_yield_1) * inputs.maturity
    )
    forward_2 = (
        inputs.hedge_ratio
        * inputs.spot_2
        * exp((inputs.risk_free_rate - inputs.dividend_yield_2) * inputs.maturity)
    )
    denominator = forward_2 + inputs.strike
    weight = forward_2 / denominator
    effective_volatility = sqrt(
        inputs.volatility_1**2
        + weight**2 * inputs.volatility_2**2
        - 2.0 * inputs.correlation * inputs.volatility_1 * inputs.volatility_2 * weight
    )
    if effective_volatility == 0:
        return exp(-inputs.risk_free_rate * inputs.maturity) * max(
            forward_1 - denominator, 0.0
        )
    root_t = sqrt(inputs.maturity)
    d_1 = (
        log(forward_1 / denominator) + 0.5 * effective_volatility**2 * inputs.maturity
    ) / (effective_volatility * root_t)
    d_2 = d_1 - effective_volatility * root_t
    return exp(-inputs.risk_free_rate * inputs.maturity) * (
        forward_1 * normal_cdf(d_1) - denominator * normal_cdf(d_2)
    )


def bump_and_revalue(
    inputs: MarketInputs,
    field: str,
    bump: float,
    paths: int = 500_000,
    seed: int = 20260903,
) -> float:
    """Central-difference sensitivity using common random numbers.

    Same draws for the up and down re-pricing: shared noise cancels in the
    difference, so the bump estimate is not drowned by fresh MC error.
    """
    if bump <= 0:
        raise ValueError("Bump must be positive.")
    values = inputs.__dict__.copy()
    if field not in values:
        raise ValueError(f"Unknown input field: {field}")
    up_values = values.copy()
    down_values = values.copy()
    up_values[field] = values[field] + bump
    down_values[field] = values[field] - bump
    up = MarketInputs(**up_values)
    down = MarketInputs(**down_values)
    up.validate()
    down.validate()

    rng = np.random.default_rng(seed)
    z_1 = rng.standard_normal(paths)
    independent = rng.standard_normal(paths)

    def price_with_correlation(candidate: MarketInputs) -> float:
        z_2 = (
            candidate.correlation * z_1
            + sqrt(max(0.0, 1.0 - candidate.correlation**2)) * independent
        )
        return float(discounted_payoffs(candidate, z_1, z_2).mean())

    return (price_with_correlation(up) - price_with_correlation(down)) / (2.0 * bump)


def conditional_discounted_payoff(inputs: MarketInputs, z_2: np.ndarray) -> np.ndarray:
    """E[discounted spread payoff | Z2=z], integrating S1 analytically.

    Conditional log(S1_T) is normal. Its mean in LEVELS is m1(z),
    its log standard deviation is v, and its strike is h*S2_T(z)+K.
    The Black--Scholes/lognormal call expectation is m1*N(d1)-k*N(d2).

    Integrating out S1 instead of simulating it is the whole trick:
    one fewer noisy dimension, and the variance reduction pays ~3000x.
    """
    inputs.validate()
    t, rho = inputs.maturity, inputs.correlation
    z_2 = np.asarray(z_2, dtype=float)
    m1 = inputs.spot_1 * np.exp(
        (
            inputs.risk_free_rate
            - inputs.dividend_yield_1
            - 0.5 * inputs.volatility_1**2 * rho**2
        )
        * t
        + inputs.volatility_1 * sqrt(t) * rho * z_2
    )
    s2 = inputs.spot_2 * np.exp(
        (inputs.risk_free_rate - inputs.dividend_yield_2 - 0.5 * inputs.volatility_2**2)
        * t
        + inputs.volatility_2 * sqrt(t) * z_2
    )
    k = inputs.hedge_ratio * s2 + inputs.strike
    v = inputs.volatility_1 * sqrt(t) * sqrt(max(0.0, 1 - rho * rho))
    if v < 1e-14:
        value = np.maximum(m1 - k, 0.0)
    else:
        d1 = (np.log(m1 / k) + 0.5 * v * v) / v
        value = m1 * ndtr(d1) - k * ndtr(d1 - v)
    return exp(-inputs.risk_free_rate * t) * value


def exact_conditional_price(inputs: MarketInputs, nodes: int = 128) -> float:
    """One-dimensional Gauss--Hermite integral; exact-model numerical benchmark."""
    if nodes < 100:
        raise ValueError("Use at least 100 Gauss-Hermite nodes")
    abscissae, weights = roots_hermitenorm(nodes)
    return float(
        weights @ conditional_discounted_payoff(inputs, abscissae) / sqrt(2 * np.pi)
    )


def margrabe_price(inputs: MarketInputs) -> float:
    """Analytical K=0 exchange-option identity, used only for validation."""
    inputs.validate()
    if inputs.strike != 0:
        raise ValueError("Margrabe identity requires K=0")
    a = inputs.spot_1 * exp(-inputs.dividend_yield_1 * inputs.maturity)
    b = (
        inputs.hedge_ratio
        * inputs.spot_2
        * exp(-inputs.dividend_yield_2 * inputs.maturity)
    )
    v = sqrt(
        max(
            0.0,
            inputs.volatility_1**2
            + inputs.volatility_2**2
            - 2 * inputs.correlation * inputs.volatility_1 * inputs.volatility_2,
        )
    ) * sqrt(inputs.maturity)
    if v < 1e-14:
        return max(a - b, 0.0)
    d1 = (log(a / b) + 0.5 * v * v) / v
    return a * normal_cdf(d1) - b * normal_cdf(d1 - v)


def variance_reduced_price(
    inputs: MarketInputs, method: str, paths: int = 500_000, seed: int = 20260903
) -> tuple[MonteCarloResult, dict]:
    """Separate estimators. Antithetic SE uses independent PAIR averages.

    Control coefficients are estimated on an independent pilot, avoiding
    same-sample fitted-control bias. Both controls have known expectations.
    """
    if paths < 2:
        raise ValueError("At least two paths are required")
    inputs.validate()
    details = {"payoff_evaluations": paths, "independent_samples": paths}
    if method == "antithetic":
        if paths % 2:
            raise ValueError("Antithetic paths must be even")
        z1, z2 = correlated_normals(paths // 2, inputs.correlation, seed)
        samples = 0.5 * (
            discounted_payoffs(inputs, z1, z2) + discounted_payoffs(inputs, -z1, -z2)
        )
        details["independent_samples"] = paths // 2
    elif method == "control_variate":
        pilot_paths = 10_000
        expected = np.array(
            [
                inputs.spot_1 * exp(-inputs.dividend_yield_1 * inputs.maturity),
                inputs.hedge_ratio
                * inputs.spot_2
                * exp(-inputs.dividend_yield_2 * inputs.maturity),
            ]
        )

        def controls(a, b):
            s1, s2 = terminal_prices(inputs, a, b)
            return exp(-inputs.risk_free_rate * inputs.maturity) * np.column_stack(
                [s1, inputs.hedge_ratio * s2]
            )

        p1, p2 = correlated_normals(pilot_paths, inputs.correlation, seed + 1)
        pc = controls(p1, p2)
        py = discounted_payoffs(inputs, p1, p2)
        coefficients, *_ = np.linalg.lstsq(
            pc - pc.mean(axis=0), py - py.mean(), rcond=None
        )
        z1, z2 = correlated_normals(paths, inputs.correlation, seed)
        samples = (
            discounted_payoffs(inputs, z1, z2)
            - (controls(z1, z2) - expected) @ coefficients
        )
        details.update(
            pilot_paths=pilot_paths,
            payoff_evaluations=paths + pilot_paths,
            coefficients=coefficients.tolist(),
            analytic_control_expectations=expected.tolist(),
        )
    elif method == "conditional_mc":
        z2 = np.random.default_rng(seed).standard_normal(paths)
        samples = conditional_discounted_payoff(inputs, z2)
    else:
        raise ValueError(f"Unknown variance-reduction method: {method}")
    result = summarize_payoffs(samples, len(samples), seed)
    # paths describes the compute budget; SE above uses independent sample count.
    return replace(result, paths=paths), details


def pathwise_greeks(
    inputs: MarketInputs, paths: int = 500_000, seed: int = 20260903
) -> dict:
    """Differentiate terminal payoff pathwise, holding contract h fixed.

    h is a term of the contract, not a hedge weight, so a spot bump must not
    move it. Vega here is per 1.00 absolute volatility; callers convert to per 1pct.
    """
    z1, z2 = correlated_normals(paths, inputs.correlation, seed)
    s1, s2 = terminal_prices(inputs, z1, z2)
    active_discount = exp(-inputs.risk_free_rate * inputs.maturity) * (
        s1 - inputs.hedge_ratio * s2 - inputs.strike > 0
    )
    derivatives = {
        "spot_1": active_discount * s1 / inputs.spot_1,
        "spot_2": -active_discount * inputs.hedge_ratio * s2 / inputs.spot_2,
        "volatility_1": active_discount
        * s1
        * (sqrt(inputs.maturity) * z1 - inputs.volatility_1 * inputs.maturity),
        "volatility_2": -active_discount
        * inputs.hedge_ratio
        * s2
        * (sqrt(inputs.maturity) * z2 - inputs.volatility_2 * inputs.maturity),
    }
    return {
        name: {
            "estimate": float(values.mean()),
            "standard_error": float(values.std(ddof=1) / sqrt(paths)),
        }
        for name, values in derivatives.items()
    }


def exact_finite_difference(
    inputs: MarketInputs, field: str, bump: float, nodes: int = 128
) -> float:
    if bump <= 0:
        raise ValueError("Bump must be positive")
    up = replace(inputs, **{field: getattr(inputs, field) + bump})
    down = replace(inputs, **{field: getattr(inputs, field) - bump})
    return (
        exact_conditional_price(up, nodes) - exact_conditional_price(down, nodes)
    ) / (2 * bump)


def adaptive_conditional_price(inputs: MarketInputs) -> tuple[float, float]:
    """Independent numerical quadrature check of the conditional integral.

    Integrate on +/-12 standard deviations; omitted normal tail mass is
    negligible here. Gauss--Hermite remains the main model benchmark.
    """

    def integrand(z):
        return (
            float(conditional_discounted_payoff(inputs, z))
            * exp(-0.5 * z * z)
            / sqrt(2 * pi)
        )

    price, error = quad(integrand, -12, 12, epsabs=1e-11, epsrel=1e-11)
    return float(price), float(error)
