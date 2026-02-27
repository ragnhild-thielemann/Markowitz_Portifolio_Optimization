"""
This file is for obtaining and cleaning data. 

We use yfinance to get open sourse stock data to later analyze

standard usage:
prices, report = fetch_prices(tickers, years=3)
prices_clean, clean_report = clean_prices(prices, min_obs=252)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Optional, Tuple, Dict, List, Union

import numpy as np
import pandas as pd
import yfinance as yf


def _as_list(x: Union[str, Iterable[str]]) -> List[str]:
    if isinstance(x, str):
        return [x]
    return [t.strip() for t in x if str(t).strip()]


def _end_as_timestamp(end: Optional[Union[str, date, datetime]]) -> pd.Timestamp:
    if end is None:
        return pd.Timestamp.today().normalize()
    return pd.to_datetime(end).normalize()


def _start_from_years(end_ts: pd.Timestamp, years: int) -> pd.Timestamp:
    days = int(round(years * 365.25))
    return (end_ts - pd.Timedelta(days=days)).normalize()

# Public API

@dataclass(frozen=True)
class FetchReport:
    requested_tickers: List[str]
    used_tickers: List[str]
    dropped_tickers: List[str]
    start: str
    end: str
    interval: str


def fetch_prices(
    tickers: Union[str, Iterable[str]],
    start: Optional[Union[str, date, datetime]] = None,
    end: Optional[Union[str, date, datetime]] = None,
    years: Optional[int] = 3,
    interval: str = "1d",
    auto_adjust: bool = False,
    progress: bool = False,
) -> Tuple[pd.DataFrame, FetchReport]:
    """
    Hent prisdata for tickere fra yfinance og returner en DataFrame med priser.

    Fetches:
        price data for tickers
    
    Returns: 
        prices: DataFrame (index: dato, columns: tickere)
        Report
    

    Std: 3 years
    Uses "adj close" if available. 

    """
    tickers_list = _as_list(tickers)
    if len(tickers_list) == 0:
        raise ValueError("No tickers chosen.")

    end_ts = _end_as_timestamp(end)

    if start is None:
        if years is None:
            raise ValueError("insert start or years!")
        start_ts = _start_from_years(end_ts, years)
    else:
        start_ts = pd.to_datetime(start).normalize()

    start_str = start_ts.strftime("%Y-%m-%d")
    end_str = end_ts.strftime("%Y-%m-%d")

    # batch download way faster.
    raw = yf.download(
        tickers = tickers_list,
        start = start_str,
        end = end_str,
        interval = interval,
        auto_adjust = auto_adjust,
        group_by = "column",
        progress = progress,
        threads = True,
    )

    # if N/As for tickers
    if raw is None or len(raw) == 0:
        report = FetchReport(
            requested_tickers=tickers_list,
            used_tickers=[],
            dropped_tickers=tickers_list,
            start=start_str,
            end=end_str,
            interval=interval,
        )
        return pd.DataFrame(), report

    prices = None

    if isinstance(raw.columns, pd.MultiIndex):
        # MultiIndex: (field, ticker) or (ticker, field) ?
        fields = list(raw.columns.get_level_values(0).unique())

        if "Adj Close" in fields:
            prices = raw["Adj Close"].copy()
        elif "Close" in fields:
            prices = raw["Close"].copy()
        else:
            raise ValueError(f"Couldn't find Adj close or close in yfinance output.")
    else:
        # Single ticker
        if "Adj Close" in raw.columns:
            prices = raw["Adj Close"].to_frame()
            prices.columns = tickers_list[:1]
        elif "Close" in raw.columns:
            prices = raw["Close"].to_frame()
            prices.columns = tickers_list[:1]
        else:
            raise ValueError("Couldn't find Adj close or close in yfinance output (single ticker).")

    # Drop tickere som er helt tomme
    all_nan_cols = [c for c in prices.columns if prices[c].isna().all()]
    prices = prices.drop(columns=all_nan_cols, errors="ignore")

    used = [c for c in tickers_list if c in prices.columns]
    dropped = [t for t in tickers_list if t not in used] + all_nan_cols

    report = FetchReport(
        requested_tickers = tickers_list,
        used_tickers = sorted(list(dict.fromkeys(used))), 
        dropped_tickers = sorted(list(dict.fromkeys(dropped))),
        start = start_str,
        end = end_str,
        interval = interval,
    )

    prices = prices.apply(pd.to_numeric, errors="coerce")

    # Normalize indexes
    prices.index = pd.to_datetime(prices.index)

    return prices, report

# ----- Clean data ------

@dataclass(frozen=True)
class CleanReport:
    input_shape: Tuple[int, int]
    output_shape: Tuple[int, int]
    dropped_tickers: List[str]
    dropped_rows: int
    method: str
    min_obs: int


def clean_prices(
    prices: pd.DataFrame,
    method: str = "dropna",
    min_obs: int = 252,
    drop_invalid: bool = True,
) -> Tuple[pd.DataFrame, CleanReport]:
    """
    Cleans data, making it ready for analysis. 

    method:
        "dropna": keeping only dates where all tickers have price

        "ffill": Skip eventual  NaNs

    min_obs:
        252 days is ca. 1 year. Minimum amount of rows per ticker

    drop_invalid:
      Skips tickers which still has too little data. 
    """
    if prices is None or prices.empty:
        report = CleanReport(
            input_shape = (0, 0),
            output_shape = (0, 0),
            dropped_tickers = [],
            dropped_rows = 0,
            method = method,
            min_obs = min_obs,
        )
        return pd.DataFrame(), report

    df = prices.copy()
    df = df.sort_index()

    input_shape = df.shape

    # removes duplicate dates.
    df = df[~df.index.duplicated(keep="last")]

    # clean stretegy: two possabilities. 
    if method == "dropna":
        before_rows = len(df)
        df = df.dropna(axis=0, how="any")
        dropped_rows = before_rows - len(df)
    elif method == "ffill":
        before_rows = len(df)
        df = df.ffill().dropna(axis=0, how="any")
        dropped_rows = before_rows - len(df)
    else:
        raise ValueError("Method has to be 'dropna' or 'ffill'")

    dropped_tickers: List[str] = []

    if drop_invalid:
        # Skips tickers with too little data
        too_few = [c for c in df.columns if df[c].notna().sum() < min_obs]
        all_nan = [c for c in df.columns if df[c].isna().all()]
        to_drop = sorted(list(set(too_few + all_nan)))

        df = df.drop(columns=to_drop, errors="ignore")
        dropped_tickers = to_drop

        # secure no N/As in the rows. 
        df = df.dropna(axis=0, how="any")

    report = CleanReport(
        input_shape=input_shape,
        output_shape=df.shape,
        dropped_tickers=dropped_tickers,
        dropped_rows=dropped_rows,
        method=method,
        min_obs=min_obs,
    )
    return df, report

