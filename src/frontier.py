"""
frontier.py

Efficient frontier analytical. 


No constraints like w_i >= 0 (shorting allowed). 
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FrontierPoint:
    mu_target: float
    port_return: float
    port_vol: float
    weights: pd.Series


def _align(mu: pd.Series, cov: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    if not isinstance(mu, pd.Series):
        raise TypeError("mu must be a pandas Series")
    if not isinstance(cov, pd.DataFrame):
        raise TypeError("cov must be a pandas DataFrame")

    assets = list(mu.index)
    missing = [a for a in assets if a not in cov.index or a not in cov.columns]
    if missing:
        raise ValueError(f"cov is missing assets: {missing}")

    cov_aligned = cov.loc[assets, assets]
    mu_vec = mu.loc[assets].to_numpy(dtype=float)
    cov_mat = cov_aligned.to_numpy(dtype=float)
    return mu_vec, cov_mat, assets


def efficient_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    n_points: int = 50,
    mu_min: float | None = None,
    mu_max: float | None = None,
    ridge: float = 0.0,
) -> Tuple[pd.DataFrame, list[pd.Series]]:
    """
    Compute the return-vol pairs and the corresponding weights.

    Parameters:
      mu: expected returns (annualized)
      cov: covariance matrix (annualized)
      n: number of points on the frontier
      mu_min, mu_max: target return range (annualized). If None, use min/max of mu.
      ridge: optional diagonal regularization added to cov.

    Returns:
      frontier_df: DataFrame with columns ["mu_target", "ret", "vol"]
      weights_list: list of pd.Series weights for each point (same order as frontier_df)
    """
    mu_vec, S, assets = _align(mu, cov)
    n = len(assets)
    ones = np.ones(n, dtype=float)

    if ridge > 0.0:
        S = S + ridge * np.eye(n)

    Sinv_ones = np.linalg.solve(S, ones)
    Sinv_mu = np.linalg.solve(S, mu_vec)

    # The scalars used in Markowitz algebra
    A = ones @ Sinv_ones
    B = ones @ Sinv_mu
    C = mu_vec @ Sinv_mu

    D = A * C - B * B
    if D == 0.0:
        raise ValueError("Cannot compute frontier (matrix is singular or degenerate). Try ridge>0.")

    if mu_min is None:
        mu_min = float(mu.min())
    if mu_max is None:
        mu_max = float(mu.max())
    if n_points < 2:
        raise ValueError("n_points must be at least 2")

    targets = np.linspace(mu_min, mu_max, n_points)

    rows = []
    weights_list: list[pd.Series] = []

    for mu_t in targets:
        # Lagrange multipliers
        lam1 = (C - B * mu_t) / D
        lam2 = (A * mu_t - B) / D

        # Optimal weights for target return
        w = lam1 * Sinv_ones + lam2 * Sinv_mu 

        # Portfolio return and volatility
        port_ret = float(w @ mu_vec)
        port_var = float(w @ S @ w)
        port_vol = float(np.sqrt(max(port_var, 0.0)))

        w_ser = pd.Series(w, index=assets, name="weight")
        weights_list.append(w_ser)

        rows.append({"mu_target": float(mu_t), "ret": port_ret, "vol": port_vol})

    frontier_df = pd.DataFrame(rows)
    return frontier_df, weights_list