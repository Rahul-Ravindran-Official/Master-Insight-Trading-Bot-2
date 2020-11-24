from typing import List, Tuple

import pandas as pd

from indicators.Indicator import Indicator
from indicators.ma.ma_type import MAType


class MovingAverage(Indicator):

    ohlc_df: pd.DataFrame
    period: int
    type: MAType

    def __init__(self, ohlc_df: pd.DataFrame, period: int = 12, ma_type: MAType = MAType.ema):
        """
        :param ohlc_df: Dataframe containing security data
        :param period: period to take the moving avg for.
        :param ma_type: Exponential | Simple
        """
        self.ohlc_df = ohlc_df.__deepcopy__()
        self.period = period
        self.type = ma_type

    def get_signal(
            self
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Calculates and returns moving average

        :return: Complete Dataframe and "ma_period_<the-period>"
        """

        signal_col_name = "ma_period_" + str(self.period)

        # Intermediary Computations - Stage 1
        if self.type == MAType.ema:
            self.ohlc_df[signal_col_name] = self.ohlc_df["Adj Close"].ewm(
                span=self.period,
                min_periods=self.period
            ).mean()
        else:
            self.ohlc_df[signal_col_name] = self.ohlc_df["Adj Close"].rolling(
                window=self.period
            ).mean()

        return self.ohlc_df, [signal_col_name]
