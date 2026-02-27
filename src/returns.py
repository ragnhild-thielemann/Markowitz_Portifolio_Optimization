"""
returns.py

Compute asset returns and estimate mean returns and covariance
for use in Markowitz portfolio optimization.
"""

from typing import Tuple

import numpy as np
import pandas as pd


def compute_returns(
    prices: pd.DataFrame,
    kind: str = "log",
) -> pd.DataFrame:
    """
    Compute stock returns from price data.

    Parameters:
      prices:
        DataFrame with dates as index and tickers as columns.

      kind:
        - log returns
        - simple percentage returns

    Returns:
      DataFrame of returns with the same columns as prices.
    """
    if prices is None or prices.empty:
        return pd.DataFrame()

    if kind == "log":
        returns = np.log(prices / prices.shift(1))
    elif kind == "simple":
        returns = prices.pct_change()
    else:
        raise ValueError("kind must be 'log' or 'simple'.")

    return returns.dropna()


def estimate_mean_returns(
    returns: pd.DataFrame,
    periods: int = 252,
) -> pd.Series:
    """
    Estimate expected mean returns.

    Parameters:
      returns:
        DataFrame of stock returns.

      periods:
        Number of periods per year (252 days in a year)

    Returns:
      Series of annualized mean returns.
    """
    if returns is None or returns.empty:
        return pd.Series(dtype=float)

    return returns.mean() * periods


def estimate_covariance(
    returns: pd.DataFrame,
    periods: int = 252,
) -> pd.DataFrame:
    """
    Estimate the covariance matrix of returns.

    Parameters:
      returns:
        DataFrame of asset returns.

      periods:
        Number of periods per year (252 for daily data).

    Returns:
      Annualized covariance matrix.
    """
    if returns is None or returns.empty:
        return pd.DataFrame()

    return returns.cov() * periods