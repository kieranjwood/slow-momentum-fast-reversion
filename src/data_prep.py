import numpy as np
import pandas as pd


def calc_returns(srs: pd.Series, offset: int = 1) -> pd.Series:
    """for each element of a pandas time-series srs,
    calculates the returns over the past number of days
    specified by offset

    Args:
        srs (pd.Series): time-series of prices
        offset (int, optional): number of days to calculate returns over. Defaults to 1.

    Returns:
        pd.Series: series of returns
    """
    returns = srs / srs.shift(offset) - 1.0
    return returns


def read_changepoint_results_and_fill_na(
    file_path: str, lookback_window_length: int
) -> pd.DataFrame:
    """Read output data from changepoint detection module into a dataframe.
    For rows where the module failed, information for changepoint location and severity is
    filled using the previous row.


    Args:
        file_path (str): the file path of the csv containing the results
        lookback_window_length (int): lookback window length - necessary for filling in the blanks for norm location

    Returns:
        pd.DataFrame: changepoint severity and location information
    """

    return (
        pd.read_csv(file_path, index_col=0, parse_dates=True)
        .fillna(method="ffill")
        .dropna()  # if first values are na
        .assign(
            cp_location_norm=lambda row: (row["t"] - row["cp_location"])
            / lookback_window_length
        )  # fill by assigning the previous cp and score, then recalculate norm location
    )
