"""
plotting.py

Plots for the efficient frontier.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_frontier(
    frontier: pd.DataFrame,
    mu: Optional[pd.Series] = None,
    cov: Optional[pd.DataFrame] = None,
    min_var: Optional[tuple[float, float]] = None,    
    max_sharpe: Optional[tuple[float, float]] = None,  
    rf: float = 0.0,
    title: str = "Efficient Frontier",
    show: bool = True,
    savepath: Optional[str] = None,
) -> None:
    """
    Plot efficient frontier (volatility on x-axis, return on y-axis).

    frontier must contain columns: "vol", "ret"
    If mu and cov are provided, plots each individual asset with ticker labels.
    If min_var/max_sharpe are provided, marks those points.
    If max_sharpe is provided, draws the Capital Market Line (CML).
    """
    required = {"vol", "ret"}
    if not required.issubset(frontier.columns):
        raise ValueError(f"frontier must contain columns {required}")

    x = frontier["vol"].to_numpy()
    y = frontier["ret"].to_numpy()

    plt.figure()
    plt.plot(x, y, marker="o", linestyle="-", label="Efficient frontier")
    plt.xlabel("Volatility (annualized)")
    plt.ylabel("Expected return (annualized)")
    plt.title(title)
    plt.grid(True)

    # Mark minimum-variance point
    if min_var is not None:
        vol_min, ret_min = min_var
        plt.scatter(
            vol_min,
            ret_min,
            marker="o",
            s=120,
            edgecolors="black",
            label="Minimum variance",
        )

    # Mark maximum sharpe point
    if max_sharpe is not None:
        vol_tan, ret_tan = max_sharpe
        plt.scatter(
            vol_tan,
            ret_tan,
            marker="*",
            s=180,
            edgecolors="black",
            label="Maximum Sharpe",
        )

        # Capital Market Line 
        if vol_tan == 0.0:
            raise ValueError("max_sharpe volatility is zero; cannot draw CML.")
        sr = (ret_tan - rf) / vol_tan
        x_cml = np.linspace(0.0, float(frontier["vol"].max()), 100)
        y_cml = rf + sr * x_cml
        plt.plot(x_cml, y_cml, linestyle="--", label="Capital Market Line")

    # Plot individual stocks (tickers)
    if mu is not None and cov is not None:
        assets = list(mu.index)
        cov_aligned = cov.loc[assets, assets]

        vols = np.sqrt(np.diag(cov_aligned.to_numpy(dtype=float)))
        rets = mu.to_numpy(dtype=float)

        plt.scatter(vols, rets, marker="x", label="Individual assets")

        for a, vx, vy in zip(assets, vols, rets):
            plt.annotate(
                str(a),
                (vx, vy),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
            )

    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight")

    if show:
        plt.legend()
        plt.show()
    else:
        plt.close()