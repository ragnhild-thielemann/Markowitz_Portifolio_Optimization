"""
markowitz.py

Analytical Markowitz portfolios:
- Minimum variance portfolio
- Maximum Sharpe portfolio

Assumptions:
- No constraints like w_i >= 0 (short selling allowed)
- Covariance matrix is invertible, at least numerically. 
"""

from typing import Union

import numpy as np
import pandas as pd


def align_inputs(mu: pd.Series, cov: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Align mu and cov to the same asset order and return numpy arrays.
    """
    if not isinstance(mu, pd.Series):
        raise TypeError("mu must be a pandas Series")
    if not isinstance(cov, pd.DataFrame):
        raise TypeError("cov must be a pandas DataFrame")

    assets = list(mu.index)

    # Ensure cov contains all assets in mu
    missing = [a for a in assets if a not in cov.index or a not in cov.columns]
    if missing:
        raise ValueError(f"Cov is missing assets.")

    cov_aligned = cov.loc[assets, assets]
    mu_vec = mu.loc[assets].to_numpy(dtype = float)
    cov_mat = cov_aligned.to_numpy(dtype = float)

    return mu_vec, cov_mat, assets


def minimum_variance_weights(
    cov: pd.DataFrame,
    ridge: float = 0.0,
) -> pd.Series:
    """
    Minimum variance portfolio. (calculations in README.md)

    Parameters:
      cov: covariance matrix (NxN) as DataFrame (index/columns = asset names)
      ridge: optional diagonal regularization (adds ridge * I)

    Returns:
      weights as pd.Series summing to 1
    """
    if not isinstance(cov, pd.DataFrame):
        raise TypeError("cov must be a pandas DataFrame")
    assets = list(cov.columns)

    S = cov.to_numpy(dtype=float)
    if ridge > 0.0:
        S = S + ridge * np.eye(S.shape[0])

    ones = np.ones(len(assets), dtype=float)

    x = np.linalg.solve(S, ones) 
    denom = ones @ x              

    if denom == 0.0:
        raise ValueError("Cannot compute minimum variance weights (denominator is zero).")

    w = x / denom
    return pd.Series(w, index = assets, name = "weight")


def max_sharpe_weights(
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float = 0.0,
    ridge: float = 0.0,
) -> pd.Series:
    """
    Maximum Sharpe portfolio with risk-free rate r_f.

    Parameters:
      mu: expected returns (annualized) as Series
      cov: covariance matrix (annualized) as DataFrame
      rf: risk-free rate (annualized), same units as mu
      ridge: optional diagonal regularization (adds ridge * I)

    Returns:
      weights as pd.Series summing to 1
    """
    mu_vec, S, assets = align_inputs(mu, cov)

    if ridge > 0.0:
        S = S + ridge * np.eye(S.shape[0])

    ones = np.ones(len(assets), dtype=float)
    excess = mu_vec - rf * ones  

    x = np.linalg.solve(S, excess) 
    denom = ones @ x

    if denom == 0.0:
        raise ValueError("Cannot normalize tangency portfolio (sum is zero).")

    w = x / denom
    return pd.Series(w, index = assets, name="weight")