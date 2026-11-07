"""Independent deterministic benchmark for the pathwise Greeks."""

import pytest
from spread_option_pricer import pathwise_greeks, exact_finite_difference


@pytest.mark.parametrize("field", ["spot_1", "spot_2", "volatility_1", "volatility_2"])
def test_pathwise_greek_within_three_own_standard_errors_of_exact_fd(inputs, field):
    estimate = pathwise_greeks(inputs)[field]
    bump = getattr(inputs, field) * 1e-4 if field.startswith("spot") else 1e-4
    exact = exact_finite_difference(inputs, field, bump)
    assert abs(estimate["estimate"] - exact) <= 3 * estimate["standard_error"]
