import os
import datetime as dt
import yfinance as yf
import src.changepoint_detection as cpd

from src.data_prep import calc_returns

TICKER = "^FTSE"
START_DATE = dt.datetime(2005, 1, 1)
END_DATE = dt.datetime(2009, 12, 31)
USE_KM_HYP_TO_INITIALISE_KC = True


data_source = yf.Ticker(TICKER)
data = data_source.history(period='max')
data["daily_returns"] = calc_returns(data["Close"])

cpd.run_module(data, 20, os.path.join("data", "test.csv"), START_DATE, END_DATE, USE_KM_HYP_TO_INITIALISE_KC)
pass
