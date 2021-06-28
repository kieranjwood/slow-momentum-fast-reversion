import numpy as np
import pandas as pd

def calc_returns(srs: pd.Series, offset:int =1) -> pd.Series:
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