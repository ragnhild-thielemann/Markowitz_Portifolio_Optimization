"""
test_returns.py

Simple tests for functions in src.returns.

Run with:
    python -m tests.test_returns
"""

import pandas as pd
import numpy as np

from src.returns import compute_returns, estimate_mean_returns, estimate_covariance

def sample_prices():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    return pd.DataFrame(
        {
            "A": [100, 101, 102, 103],
            "B": [200, 198, 202, 204],
        },
        index = dates,
    )


def test_compute_returns():
    prices = sample_prices()
    returns = compute_returns(prices, kind="log")

    assert not returns.empty
    assert returns.shape[1] == 2
    assert isinstance(returns.iloc[0, 0], float)

    print("test_compute_returns passed")


def test_estimate_mean_returns():
    prices = sample_prices()
    returns = compute_returns(prices)
    mu = estimate_mean_returns(returns, periods = 252)

    assert isinstance(mu, pd.Series)
    assert set(mu.index) == {"A", "B"}

    print("test_estimate_mean_returns passed")


def test_estimate_covariance():
    prices = sample_prices()
    returns = compute_returns(prices)
    cov = estimate_covariance(returns, periods=252)

    assert isinstance(cov, pd.DataFrame)
    assert cov.shape == (2, 2)
    assert np.allclose(cov, cov.T)  # covariance must be symmetric

    print("test_estimate_covariance passed")


def run_all_tests():
    test_compute_returns()
    test_estimate_mean_returns()
    test_estimate_covariance()
    print("All returns tests passed!")


if __name__ == "__main__":
    run_all_tests()