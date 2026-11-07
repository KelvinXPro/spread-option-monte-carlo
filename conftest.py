"""Price-panel gating for the TSLA/TSM spread-option tests.

The calibration panel is a Bloomberg extract and is not redistributed.
run_analysis.load_inputs falls back to the calibrated contract inputs in
results/summary.json when the panel is absent, so every pricing, quadrature
and Greek check still runs. Only tests that inspect the price panel itself
skip.
"""
import pytest

from run_analysis import DATA_PATH, load_inputs


@pytest.fixture(scope="module")
def inputs():
    return load_inputs()[1]


@pytest.fixture(scope="module")
def price_panel():
    if not DATA_PATH.is_file():
        pytest.skip(
            "Bloomberg price panel not present (not redistributable); "
            "place data/tsla_tsm_prices_2020_2025.csv to run this test"
        )
    return load_inputs()[0]
