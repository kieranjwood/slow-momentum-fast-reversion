import numpy as np
import pandas as pd


def macd_signal(prices, short_window, long_window, x_vol_window=63, y_vol_window=252):
    def get_halflife(n):
        return np.log(0.5) / np.log(1 - 1 / n)

    # Compute
    short_trend = prices.ewm(halflife=get_halflife(short_window)).mean()
    long_trend = prices.ewm(halflife=get_halflife(long_window)).mean()

    x = short_trend - long_trend

    y = x / prices.rolling(x_vol_window).std().fillna(method="bfill")

    z = y / y.rolling(y_vol_window).std().fillna(method="bfill")

    return z


class MACDStrategy:
    def __init__(self, trend_combinations=None, x_vol_window=63, y_vol_window=252):

        self.x_vol_window = x_vol_window
        self.y_vol_window = y_vol_window

        if trend_combinations is None:
            self.trend_combinations = [(8, 24), (16, 48), (32, 96)]
        else:
            self.trend_combinations = trend_combinations

    def combined_signal(self, prices):
        def scale_signal(z):

            return z * np.exp(-(z ** 2) / 4) / 0.89

        trend_combinations = self.trend_combinations
        signal_df = None
        for short_window, long_window in trend_combinations:

            indiv_signal = macd_signal(
                prices, short_window, long_window, self.x_vol_window, self.y_vol_window
            )

            if signal_df is None:
                signal_df = scale_signal(indiv_signal)
            else:
                signal_df += scale_signal(indiv_signal)

        return signal_df / len(trend_combinations)