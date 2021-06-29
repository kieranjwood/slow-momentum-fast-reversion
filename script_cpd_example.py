import argparse
import datetime as dt

import pandas as pd
import yfinance as yf

import src.changepoint_detection as cpd
from src.data_prep import calc_returns

USE_KM_HYP_TO_INITIALISE_KC = True
LBW = 21


def main(
    ticker: str, output_file_path: str, start_date: dt.datetime, end_date: dt.datetime
):
    data_source = yf.Ticker(ticker)
    data = data_source.history(period="max")
    data["daily_returns"] = calc_returns(data["Close"])

    cpd.run_module(
        data, LBW, output_file_path, start_date, end_date, USE_KM_HYP_TO_INITIALISE_KC
    )


if __name__ == "__main__":

    def get_args():
        """Returns settings from command line."""

        parser = argparse.ArgumentParser(description="Run changepoint detection module")
        parser.add_argument(
            "ticker",
            metavar="t",
            type=str,
            nargs="?",
            default="^FTSE",
            # choices=[],
            help="Ticker type",
        )
        parser.add_argument(
            "output_file_path",
            metavar="f",
            type=str,
            nargs="?",
            default="data/test.csv",
            # choices=[],
            help="Output file location for csv.",
        )
        parser.add_argument(
            "start_date",
            metavar="s",
            type=str,
            nargs="?",
            default="2005-01-01",
            help="Start date in format yyyy-mm-dd",
        )
        parser.add_argument(
            "end_date",
            metavar="e",
            type=str,
            nargs="?",
            default="2009-12-31",
            help="End date in format yyyy-mm-dd",
        )

        args = parser.parse_known_args()[0]

        start_date = dt.datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = dt.datetime.strptime(args.end_date, "%Y-%m-%d")

        return (
            args.ticker,
            args.output_file_path,
            start_date,
            end_date,
        )

    main(*get_args())
