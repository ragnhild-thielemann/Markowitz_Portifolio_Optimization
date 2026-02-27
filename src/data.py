"""
This file is responsible for obtaining and cleaning stock price data.

We use yfinance to fetch open-source historical stock data for later analysis.

Standard usage:
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
    """
    Ensures the input tickers are returned as a list.
    
    Accepts either a single ticker string or an iterable of strings.
    Strips whitespace and ignores empty strings.
    """
    if isinstance(x, str):
        return [x]
    return [t.strip() for t in x if str(t).strip()] 


def _end_as_timestamp(end: Optional[Union[str, date, datetime]]) -> pd.Timestamp:
    """
    Converts the end date to a pandas Timestamp.
    Defaults to today's date if None is provided.
    """
    if end is None:
        return pd.Timestamp.today().normalize()
    return pd.to_datetime(end).normalize()


def _start_from_years(end_ts: pd.Timestamp, years: int) -> pd.Timestamp:
    """
    Calculates the start date based on a number of years backward from end_ts.
    
    Accounts for leap years using 365.25 days per year.
    """
    days = int(round(years * 365.25))
    return (end_ts - pd.Timedelta(days=days)).normalize()


# ---------------- Public API ----------------

@dataclass(frozen=True)
class FetchReport:
    """
    Stores a report of the fetch operation, tracking requested, used, and dropped tickers,
    as well as the date range and interval used.
    """
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
    Fetches historical price data for the specified tickers from yfinance and returns
    a DataFrame of prices along with a FetchReport.

    Parameters:
        tickers: Single ticker string or a list of ticker symbols.
        start: Optional start date. Defaults to 'years' before 'end'.
        end: Optional end date. Defaults to today if None.
        years: Number of years of data to fetch if start is None. Default is 3.
        interval: Data frequency (e.g., "1d" for daily).
        auto_adjust: Whether to adjust prices for splits/dividends.
        progress: Show progress bar while fetching data.

    Returns:
        prices: DataFrame with datetime index and tickers as columns.
        report: FetchReport summarizing the fetched data.
    """
    tickers_list = _as_list(tickers)  # Ensure tickers are in list format
    if len(tickers_list) == 0:
        raise ValueError("No tickers provided.")

    end_ts = _end_as_timestamp(end)

    if start is None:
        if years is None:
            raise ValueError("Either 'start' date or 'years' must be specified.")
        start_ts = _start_from_years(end_ts, years)
    else:
        start_ts = pd.to_datetime(start).normalize()

    start_str = start_ts.strftime("%Y-%m-%d")
    end_str = end_ts.strftime("%Y-%m-%d")  # Convert to string format required by yfinance

    # Fetch data in batch (much faster for multiple tickers)
    raw = yf.download(
        tickers=tickers_list,
        start=start_str,
        end=end_str,
        interval=interval,
        auto_adjust=auto_adjust,
        group_by="column",  # Group data by ticker
        progress=progress,
        threads=True,        # Enable multi-threading for faster download
    )

    # Return empty DataFrame if no data is retrieved
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

    # Handle multiple tickers with MultiIndex
    if isinstance(raw.columns, pd.MultiIndex):
        fields = list(raw.columns.get_level_values(0).unique())

        if "Adj Close" in fields:
            prices = raw["Adj Close"].copy()
        elif "Close" in fields:
            prices = raw["Close"].copy()
        else:
            raise ValueError("Neither 'Adj Close' nor 'Close' found in yfinance output.")
    else:
        # Single ticker case
        if "Adj Close" in raw.columns:
            prices = raw["Adj Close"].to_frame()
            prices.columns = tickers_list[:1]
        elif "Close" in raw.columns:
            prices = raw["Close"].to_frame()
            prices.columns = tickers_list[:1]
        else:
            raise ValueError("Neither 'Adj Close' nor 'Close' found for single ticker.")

    # Drop tickers with all missing values
    all_nan_cols = [c for c in prices.columns if prices[c].isna().all()]
    prices = prices.drop(columns=all_nan_cols, errors="ignore")

    # Identify used and dropped tickers
    used = [c for c in tickers_list if c in prices.columns]
    dropped = [t for t in tickers_list if t not in used] + all_nan_cols

    report = FetchReport(
        requested_tickers=tickers_list,
        used_tickers=sorted(list(dict.fromkeys(used))),
        dropped_tickers=sorted(list(dict.fromkeys(dropped))),
        start=start_str,
        end=end_str,
        interval=interval,
    )

    # Ensure all values are numeric
    prices = prices.apply(pd.to_numeric, errors="coerce")

    # Normalize index to datetime
    prices.index = pd.to_datetime(prices.index)

    return prices, report


# ---------------- Cleaning Data ----------------

@dataclass(frozen=True)
class CleanReport:
    """
    Stores a report of the cleaning process, including input/output shape,
    tickers removed, number of dropped rows, and the cleaning method applied.
    """
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
    Cleans the price data to prepare it for analysis.

    Parameters:
        prices: DataFrame of raw prices.
        method: Cleaning method. "dropna" removes rows with any missing values,
                "ffill" forward-fills missing values before removing remaining NaNs.
        min_obs: Minimum number of valid observations required per ticker (default 252 ~ 1 year).
        drop_invalid: Whether to drop tickers that do not meet the minimum observation threshold.

    Returns:
        df: Cleaned DataFrame.
        report: CleanReport summarizing the cleaning process.
    """
    if prices is None or prices.empty:
        report = CleanReport(
            input_shape=(0, 0),
            output_shape=(0, 0),
            dropped_tickers=[],
            dropped_rows=0,
            method=method,
            min_obs=min_obs,
        )
        return pd.DataFrame(), report

    df = prices.copy()
    df = df.sort_index()
    input_shape = df.shape

    # Remove duplicate dates
    df = df[~df.index.duplicated(keep="last")]

    # Apply cleaning strategy
    if method == "dropna":
        before_rows = len(df)
        df = df.dropna(axis=0, how="any")
        dropped_rows = before_rows - len(df)
    elif method == "ffill":
        before_rows = len(df)
        df = df.ffill().dropna(axis=0, how="any")
        dropped_rows = before_rows - len(df)
    else:
        raise ValueError("Method must be either 'dropna' or 'ffill'.")

    dropped_tickers: List[str] = []

    # Drop tickers with insufficient data
    if drop_invalid:
        too_few = [c for c in df.columns if df[c].notna().sum() < min_obs]
        all_nan = [c for c in df.columns if df[c].isna().all()]
        to_drop = sorted(list(set(too_few + all_nan)))

        df = df.drop(columns=to_drop, errors="ignore")
        dropped_tickers = to_drop

        # Ensure no remaining NaNs
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
