"""
test_indicators.py

Simple tests for functions in src.indicators.

Run with:
    python -m tests.test_indicators
"""

import numpy as np
import pandas as pd

from src.indicators import portfolio_return, portfolio_variance
from src.indicators import portfolio_volatility, sharpe_ratio


def sample_inputs():
    # Two-asset example
    mu = pd.Series([0.10, 0.05], index=["A", "B"])
    cov = pd.DataFrame(
        [[0.04, 0.01],
         [0.01, 0.09]],
        index=["A", "B"],
        columns=["A", "B"],
    )
    w = np.array([0.6, 0.4])
    return w, mu, cov


def test_portfolio_return():
    w, mu, _ = sample_inputs()
    rp = portfolio_return(w, mu)

    assert isinstance(rp, float)
    assert abs(rp - (0.6 * 0.10 + 0.4 * 0.05)) < 1e-12

    print("test_portfolio_return passed")


def test_portfolio_variance_and_volatility():
    w, _, cov = sample_inputs()
    var = portfolio_variance(w, cov)
    vol = portfolio_volatility(w, cov)

    assert isinstance(var, float)
    assert isinstance(vol, float)
    assert var >= 0.0
    assert abs(vol - np.sqrt(var)) < 1e-12

    print("test_portfolio_variance_and_volatility passed")


def test_sharpe_ratio():
    w, mu, cov = sample_inputs()
    sr = sharpe_ratio(w, mu, cov, rf=0.0)

    assert isinstance(sr, float)

    print("test_sharpe_ratio passed")


def run_all_tests():
    test_portfolio_return()
    test_portfolio_variance_and_volatility()
    test_sharpe_ratio()
    print("All indicators tests passed!")


if __name__ == "__main__":
    run_all_tests()