import os
from typing import List

import pandas as pd
import numpy as np


def pull_quandl_sample_data(ticker: str) -> pd.DataFrame:
    return (
        pd.read_csv(os.path.join("data", "quandl", f"{ticker}.csv"), parse_dates=[0])
        .rename(columns={"Trade Date": "date", "Date": "date", "Settle": "close"})
        .set_index("date")
        .replace(0.0, np.nan)
    )

def pull_quandl_sample_data_multiple(tickers: List[str]) -> pd.DataFrame:
    return pd.concat(
        [
            pull_quandl_sample_data(ticker).assign(ticker=ticker).copy()
            for ticker in tickers
        ]
    )
