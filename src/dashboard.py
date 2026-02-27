"""
dashboard.py

Terminal dashboard using Rich (static output).
"""
from __future__ import annotations

import pandas as pd
from typing import Optional, Dict, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.align import Align

console = Console()


def show_dashboard(
    tickers: list[str],
    mu: pd.Series,
    w_min: pd.Series,
    w_tan: pd.Series,
    min_ret: float,
    min_vol: float,
    min_sr: float,
    tan_ret: float,
    tan_vol: float,
    tan_sr: float,
    rf: float,
    years: int,
    risk_info: Optional[Dict[str, Any]] = None
) -> None:
    table = Table(title="Markowitz Portfolio Results", show_lines=True)

    table.add_column("Asset", justify="left")
    table.add_column("μ (annual)", justify="right")
    table.add_column("Min-Var w", justify="right")
    table.add_column("Max-Sharpe w", justify="right")


    for asset in mu.index:
        table.add_row(
            str(asset),
            f"{float(mu[asset]):.4f}",
            f"{float(w_min.get(asset, 0.0)):.4f}",
            f"{float(w_tan.get(asset, 0.0)):.4f}",
        )

    header = (
        f"[bold]Tickers[/bold]: {', '.join(tickers)}\n"
        f"[bold]Years[/bold]: {years}    [bold]rf[/bold]: {rf:.4f}"
    )

    metrics = (
        f"[bold]Min-Variance[/bold]\n"
        f"Return: {min_ret:.4f}\n"
        f"Vol:    {min_vol:.4f}\n"
        f"Sharpe: {min_sr:.4f}\n\n"
        f"[bold]Max-Sharpe[/bold]\n"
        f"Return: {tan_ret:.4f}\n"
        f"Vol:    {tan_vol:.4f}\n"
        f"Sharpe: {tan_sr:.4f}"
    )

    console.print(Panel(header, title = "Run settings", border_style = "blue"))
    console.print(Panel(Align.center(table), title = "Weights + μ", border_style = "cyan"))
    console.print(Panel(metrics, title = "Metrics (annualized)", border_style = "green"))

        # Risk profile recommendation panel
    if risk_info is not None:
        w_rec = risk_info.get("w_rec", None)

        if w_rec is None:
            console.print(
                Panel(
                    "risk_info was provided, but no recommended weights (w_rec) were found.",
                    title="Risk profile recommendation",
                    border_style="magenta",
                )
            )
        else:
            w_rec = w_rec.sort_values(ascending=False)

            top_lines = []
            for asset, val in w_rec.head(6).items():
                top_lines.append(f"{asset}: {val*100:.2f}%")
            top_txt = "\n".join(top_lines) if top_lines else "No weights."

            txt = (
                f"[bold]Score:[/bold] {risk_info.get('score', 0.0):.2f}\n"
                f"[bold]Strategy:[/bold] {risk_info.get('strategy', '')}\n"
                f"{risk_info.get('explanation', '')}\n\n"
                f"[bold]Inputs[/bold]\n"
                f"Time horizon: {risk_info.get('time_horizon_years', 0.0):.1f} years\n"
                f"Loss tolerance: {risk_info.get('loss_tolerance', 0)}/5\n"
                f"Experience: {risk_info.get('experience', 0)}/5\n\n"
                f"[bold]Recommended portfolio (annualized)[/bold]\n"
                f"Return: {risk_info.get('rec_ret', 0.0):.4f}\n"
                f"Vol:    {risk_info.get('rec_vol', 0.0):.4f}\n"
                f"Sharpe: {risk_info.get('rec_sr', 0.0):.4f}\n\n"
                f"[bold]Top weights[/bold]\n{top_txt}"
            )

            console.print(
                Panel(
                    txt,
                    title="Risk profile recommendation",
                    border_style="magenta",
                )
            )