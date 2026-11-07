"""Independent exact-integral finite-difference check of the pathwise Greeks."""

from pathlib import Path
import json
import pandas as pd
from run_analysis import ROOT, load_inputs
from spread_option_pricer import pathwise_greeks, exact_finite_difference


def main():
    inputs = load_inputs()[1]
    pw = pathwise_greeks(inputs)
    rows = []
    for field in ["spot_1", "spot_2", "volatility_1", "volatility_2"]:
        bump = getattr(inputs, field) * 1e-4 if field.startswith("spot") else 1e-4
        exact = exact_finite_difference(inputs, field, bump)
        estimate = pw[field]
        rows.append(
            {
                "field": field,
                "pathwise": estimate["estimate"],
                "pathwise_standard_error": estimate["standard_error"],
                "exact_finite_difference": exact,
                "difference": estimate["estimate"] - exact,
                "difference_in_pathwise_se": (estimate["estimate"] - exact)
                / estimate["standard_error"],
                "within_3_pathwise_se": abs(estimate["estimate"] - exact)
                <= 3 * estimate["standard_error"],
                "exact_bump": bump,
            }
        )
    output = ROOT / "greek_validation"
    output.mkdir(exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "independent_greeks.csv", index=False)
    (output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
