"""
test_markowitz.py

Simple tests for functions in src.markowitz.

Run with:
    python -m tests.test_markowitz
"""

import numpy as np
import pandas as pd

from src.markowitz import minimum_variance_weights, max_sharpe_weights
from src.indicators import portfolio_volatility, sharpe_ratio


def sample_inputs():
    assets = ["A", "B", "C"]
    mu = pd.Series([0.10, 0.07, 0.04], index = assets)  

    cov = pd.DataFrame(
        [
            [0.040, 0.010, 0.000],
            [0.010, 0.090, 0.002],
            [0.000, 0.002, 0.020],
        ],
        index=assets,
        columns=assets,
    )
    return mu, cov


def test_minimum_variance_weights():
    _, cov = sample_inputs()
    w = minimum_variance_weights(cov)

    assert isinstance(w, pd.Series)
    assert abs(w.sum() - 1.0) < 1e-12

    # Min var should have <= volatility than equal weights 
    w_eq = np.ones(len(w)) / len(w)
    vol_min = portfolio_volatility(w.values, cov.values)
    vol_eq = portfolio_volatility(w_eq, cov.values)
    assert vol_min <= vol_eq + 1e-12

    print("test_minimum_variance_weights passed")


def test_max_sharpe_weights():
    mu, cov = sample_inputs()
    w = max_sharpe_weights(mu, cov, rf=0.0)

    assert isinstance(w, pd.Series)
    assert abs(w.sum() - 1.0) < 1e-12

    # Portfolio should have >= Sharpe than equal weights 
    w_eq = np.ones(len(w)) / len(w)
    sr_tan = sharpe_ratio(w.values, mu.values, cov.values, rf=0.0)
    sr_eq = sharpe_ratio(w_eq, mu.values, cov.values, rf=0.0)
    assert sr_tan >= sr_eq - 1e-12

    print("test_max_sharpe_weights passed")


def run_all_tests():
    test_minimum_variance_weights()
    test_max_sharpe_weights()
    print("All markowitz tests passed!")


if __name__ == "__main__":
    run_all_tests()