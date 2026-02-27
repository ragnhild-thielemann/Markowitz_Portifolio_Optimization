"""
risk_profile.py

Simple risk profiling:
- Ask a few questions
- Compute a risk score 0-10 
- Map score to a strategy choice

rule-based
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


Strategy = Literal["min_variance", "balanced", "max_sharpe"]


@dataclass(frozen=True)
class RiskProfile:
    time_horizon_years: float
    loss_tolerance: int      # 1–5
    experience: int          # 1–5


@dataclass(frozen=True)
class RiskDecision:
    score: float
    strategy: Strategy
    explanation: str


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


def ask_risk_profile() -> RiskProfile:
    """
    Simple CLI prompts. Keeps it minimal and demo-friendly.
    """
    time_horizon = float(input("How long until you liquidate (years)? ").strip())

    loss_tol = int(input("Loss tolerance (1=low, 5=high): ").strip())
    loss_tol = _clamp_int(loss_tol, 1, 5)

    exp = int(input("Investing experience (1=none, 5=high): ").strip())
    exp = _clamp_int(exp, 1, 5)

    return RiskProfile(
        time_horizon_years=time_horizon,
        loss_tolerance=loss_tol,
        experience=exp,
    )


def risk_score(profile: RiskProfile) -> float:
    """
    Convert answers -> risk score.

    Design choice:
    - time horizon is capped at 10 years (for scoring)
    - loss tolerance is most important
    - experience matters a bit
    """
    horizon = max(0.0, min(10.0, profile.time_horizon_years))  

    score = (
        0.4 * (horizon / 10.0) * 10.0 +     # 0..4
        1.1 * profile.loss_tolerance +      # 1.1..5.5
        0.6 * profile.experience            # 0.6..3.0
    )
    return float(score)


def choose_strategy(score: float) -> RiskDecision:
    """
    Map score -> portfolio strategy.
    """
    if score < 5.0:
        return RiskDecision(
            score=score,
            strategy="min_variance",
            explanation="Low risk profile: prioritize stability (minimum variance).",
        )
    if score < 7.5:
        return RiskDecision(
            score=score,
            strategy="balanced",
            explanation="Medium risk profile: blend between min-variance and max-Sharpe.",
        )
    return RiskDecision(
        score=score,
        strategy="max_sharpe",
        explanation="High risk profile: prioritize risk-adjusted returns (maximum Sharpe).",
    )


def blend_weights(w_min: pd.Series, w_tan: pd.Series, alpha: float) -> pd.Series:
    """
    alpha=0 -> min-variance
    alpha=1 -> max-Sharpe
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0,1]")

    # align indices
    assets = list(w_min.index)
    w_tan = w_tan.reindex(assets)

    w = (1.0 - alpha) * w_min + alpha * w_tan
    # normalize to sum to 1 (protect against rounding)
    s = float(w.sum())
    if s == 0.0:
        raise ValueError("Blended weights sum to zero (cannot normalize).")
    return (w / s).rename("weight")