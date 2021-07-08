import numpy as np
import pandas as pd

from typing import List, Tuple

VOL_LOOKBACK = 60  # for ex-ante volatility
VOL_TARGET = 0.15  # 15% volatility target


def calc_returns(srs: pd.Series, day_offset: int = 1) -> pd.Series:
    """for each element of a pandas time-series srs,
    calculates the returns over the past number of days
    specified by offset

    Args:
        srs (pd.Series): time-series of prices
        day_offset (int, optional): number of days to calculate returns over. Defaults to 1.

    Returns:
        pd.Series: series of returns
    """
    returns = srs / srs.shift(day_offset) - 1.0
    return returns


def calc_daily_vol(daily_returns):
    return (
        daily_returns.ewm(span=VOL_LOOKBACK, min_periods=VOL_LOOKBACK)
        .std()
        .fillna(method="bfill")
    )


def calc_vol_scaled_returns(daily_returns, daily_vol=None):
    """calculates volatility scaled returns for annualised VOL_TARGET of 15%
    with input of pandas series daily_returns"""
    if not daily_vol:
        daily_vol = calc_daily_vol(daily_returns)
    annualised_vol = daily_vol * np.sqrt(252)  # annualised
    return daily_returns * VOL_TARGET / annualised_vol.shift(1)


def calc_trend_intermediate_strategy(srs: pd.Series, w: float, volatility_scaling=True) -> pd.Series:
    """Calculate intermediate strategy

    Args:
        srs (pd.Series): series of prices
        w (float): weight, w=0 is Moskowitz TSMOM
        volatility_scaling (bool, optional): [description]. Defaults to True.

    Returns:
        pd.Series: series of captured returns
    """
    daily_returns = calc_returns(srs)
    monthly_returns = calc_returns(srs, 21)
    annual_returns = calc_returns(srs, 252)

    next_day_returns = (
        calc_vol_scaled_returns(daily_returns).shift(-1)
        if volatility_scaling
        else daily_returns.shift(-1)
    )

    return (
      w * np.sign(monthly_returns) * next_day_returns
      +  (1-w) * np.sign(annual_returns) * next_day_returns
  )


class MACDStrategy:
    def __init__(self, trend_combinations: List[Tuple[float, float]] = None):
        """Used to calculated the combined MACD signal for a multiple short/signal combinations,
        as described in https://arxiv.org/pdf/1904.04912.pdf

        Args:
            trend_combinations (List[Tuple[float, float]], optional): short/long trend combinations. Defaults to None.
        """
        if trend_combinations is None:
            self.trend_combinations = [(8, 24), (16, 48), (32, 96)]
        else:
            self.trend_combinations = trend_combinations

    @staticmethod
    def calc_signal(srs: pd.Series, short_timescale: int, long_timescale: int) -> float:
        """Calculate MACD signal for a signal short/long timescale combination

        Args:
            srs ([type]): series of prices
            short_timescale ([type]): short timescale
            long_timescale ([type]): long timescale

        Returns:
            float: MACD signal
        """

        def _calc_halflife(timescale):
            return np.log(0.5) / np.log(1 - 1 / timescale)

        macd = (
            srs.ewm(halflife=_calc_halflife(short_timescale)).mean()
            - srs.ewm(halflife=_calc_halflife(long_timescale)).mean()
        )
        q = macd / srs.rolling(63).std().fillna(method="bfill")
        return q / q.rolling(252).std().fillna(method="bfill")

    @staticmethod
    def scale_signal(y):
        return y * np.exp(-(y ** 2) / 4) / 0.89

    def calc_combined_signal(self, srs: pd.Series) -> float:
        """Combined MACD signal

        Args:
            srs (pd.Series): series of prices

        Returns:
            float: MACD combined signal
        """
        return np.sum(
            [self.calc_signal(srs, S, L) for S, L in self.trend_combinations]
        ) / len(self.trend_combinations)
