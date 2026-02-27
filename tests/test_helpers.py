"""
test_helpers.py

simple tests for help functons in src.data.

Run with:
python -m tests.test_helpers
"""

import pandas as pd

from src.data import _as_list, _end_as_timestamp, _start_from_years


def test_as_list():
    # string -> list
    assert _as_list("AAPL") == ["AAPL"]

    # list -> list
    assert _as_list(["AAPL", "MSFT"]) == ["AAPL", "MSFT"]

    # handle whitespace 
    assert _as_list([" AAPL ", ""]) == ["AAPL"]

    print("test_as_list passed")


def test_end_as_timestamp():
    ts = _end_as_timestamp(None)
    assert isinstance(ts, pd.Timestamp)

    ts2 = _end_as_timestamp("2024-01-01")
    assert ts2 == pd.Timestamp("2024-01-01")

    print("test_end_as_timestamp passed")


def test_start_from_years():
    end = pd.Timestamp("2024-01-01")
    start = _start_from_years(end, years=3)

    assert start < end

    # Ca. 3 years
    delta_days = (end - start).days
    assert 1090 <= delta_days <= 1100

    print("test_start_from_years passed")


def run_all_tests():
    test_as_list()
    test_end_as_timestamp()
    test_start_from_years()
    print("All test passed!")


if __name__ == "__main__":
    run_all_tests()