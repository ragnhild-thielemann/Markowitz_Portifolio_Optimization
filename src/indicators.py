"""
indicators.py

Portfolio metrics used in Markowitz portfolio optimization.
"""

from typing import Union

import numpy as np
import pandas as pd


def portfolio_return(weights: Union[pd.Series, np.ndarray], mu: pd.Series) -> float:
    """
    Expected portfolio return.

    Parameters:
      weights: portfolio weights (length N)
      mu: expected returns per stock (length N)

    Returns:
      float: expected portfolio return
    """
    w = np.asarray(weights, dtype=float)
    m = np.asarray(mu, dtype=float)
    if w.size != m.size:
        raise ValueError("weights and mu must have the same length!")
    return float(w @ m)


def portfolio_variance(
    weights: Union[pd.Series, np.ndarray],
    cov: Union[pd.DataFrame, np.ndarray],
) -> float:
    """
    Portfolio variance: w^T * Sigma * w
    """
    w = np.asarray(weights, dtype=float)
    S = np.asarray(cov, dtype=float)
    if S.shape[0] != S.shape[1]:
        raise ValueError("cov must be a square matrix")
    if w.size != S.shape[0]:
        raise ValueError("weights length must match cov dimensions")
    return float(w @ S @ w)


def portfolio_volatility(
    weights: Union[pd.Series, np.ndarray],
    cov: Union[pd.DataFrame, np.ndarray],
) -> float:
    """
    Portfolio volatility (standard deviation): sqrt(w^T * Sigma * w)
    """
    var = portfolio_variance(weights, cov)
    return float(np.sqrt(max(var, 0.0)))


def sharpe_ratio(
    weights: Union[pd.Series, np.ndarray],
    mu: pd.Series,
    cov: Union[pd.DataFrame, np.ndarray],
    rf: float = 0.0,
) -> float:
    """
    Sharpe ratio: (E[r_p] - rf) / sigma_p

    Parameters:
      rf: risk-free rate 
    """
    rp = portfolio_return(weights, mu)
    vol = portfolio_volatility(weights, cov)
    if vol == 0.0:
        raise ValueError("portfolio volatility is zero. Sharpe ratio is undefined")
    return float((rp - rf) / vol)