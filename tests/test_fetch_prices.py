"""
test_fetch_prices.py

Simple tests for fetch_prices in src.data.

Run with:
    python -m tests.test_fetch_prices
"""

import pandas as pd

from src.data import fetch_prices


def test_fetch_prices_basic():
    prices, report = fetch_prices(
        ["AAPL", "MSFT"],
        years=1,
    )

    # prices should not be empty
    assert not prices.empty

    # index should be datetime
    assert isinstance(prices.index, pd.DatetimeIndex)

    # columns should match tickers
    assert set(prices.columns) == {"AAPL", "MSFT"}

    # report should reflect used tickers
    assert set(report.used_tickers) == {"AAPL", "MSFT"}
    assert report.dropped_tickers == []

    print("test_fetch_prices_basic passed")


def run_all_tests():
    test_fetch_prices_basic()
    print("All fetch_prices tests passed!")


if __name__ == "__main__":
    run_all_tests()