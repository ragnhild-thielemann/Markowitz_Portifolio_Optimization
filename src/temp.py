"""
Just a check to see if data looks ok. Example use temporarily
"""

from src.data import fetch_prices, clean_prices

tickers = ["EQNR.OL", "AAPL", "MSFT", "ETH-USD"]
prices, fetch_report = fetch_prices(tickers, years=3)
prices, clean_report = clean_prices(prices, method="dropna", min_obs=252)

print(fetch_report)
print(clean_report)
print(prices)  # .head(180)