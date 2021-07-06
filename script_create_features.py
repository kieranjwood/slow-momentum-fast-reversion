import argparse
import datetime as dt
from typing import List

import pandas as pd

from data.pull_data import pull_yahoo_sample_data
from settings.default import (
    COMMODITIES_TICKERS,
    CPD_DEFAULT_LBW,
    CPD_OUTPUT_FOLDER_DEFAULT,
    FEATURES_FILE_PATH_DEFAULT,
)
from src.data_prep import deep_momentum_strategy_features, include_changepoint_features


def main(
    tickers: List[str],
    cpd_module_folder: str,
    lookback_window_length: int,
    output_file_path: str,
):
    features = pd.concat(
        [
            deep_momentum_strategy_features(pull_yahoo_sample_data(ticker)).assign(
                ticker=ticker
            )
            for ticker in tickers
        ]
    )

    include_changepoint_features(
        features, cpd_module_folder, lookback_window_length
    ).to_csv(output_file_path)


if __name__ == "__main__":

    def get_args():
        """Returns settings from command line."""

        parser = argparse.ArgumentParser(description="Run changepoint detection module")
        parser.add_argument(
            "cpd_module_folder",
            metavar="c",
            type=str,
            nargs="?",
            default=CPD_OUTPUT_FOLDER_DEFAULT,
            # choices=[],
            help="Input folder for CPD outputs.",
        )
        parser.add_argument(
            "lookback_window_length",
            metavar="l",
            type=int,
            nargs="?",
            default=CPD_DEFAULT_LBW,
            # choices=[],
            help="Input folder for CPD outputs.",
        )
        parser.add_argument(
            "output_file_path",
            metavar="f",
            type=str,
            nargs="?",
            default=FEATURES_FILE_PATH_DEFAULT,
            # choices=[],
            help="Output file location for csv.",
        )

        args = parser.parse_known_args()[0]

        return (
            COMMODITIES_TICKERS,
            args.cpd_module_folder,
            args.lookback_window_length,
            args.output_file_path,
        )

    main(*get_args())
