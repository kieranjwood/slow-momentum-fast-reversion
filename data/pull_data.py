from typing import List

import pandas as pd
import yfinance as yf


def pull_yahoo_sample_data(ticker: str) -> pd.DataFrame:
    data_source = yf.Ticker(ticker)
    return (
        data_source.history(period="max")[["Close"]]
        .rename(columns={"Close": "close"})
        .copy()
    )


def pull_yahoo_sample_data_multiple(tickers: List[str]) -> pd.DataFrame:
    return pd.concat(
        [
            pull_yahoo_sample_data(ticker).assign(ticker=ticker).copy()
            for ticker in tickers
        ]
    )
