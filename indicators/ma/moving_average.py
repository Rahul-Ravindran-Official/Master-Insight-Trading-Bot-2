from typing import List, Tuple

import pandas as pd

from indicators.Indicator import Indicator
from indicators.ma.ma_type import MAType


class MovingAverage(Indicator):
    period: int
    type: MAType

    def __init__(self, period: int = 12, ma_type: MAType = MAType.ema):
        self.period = period
        self.type = ma_type

    def get_signal(
            self,
            ohlc_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Calculates and returns moving average

        :param ohlc_df: A yfinance OHLC Dataframe
        :return: Complete Dataframe and "ma_period_<the-period>"
        """

        signal_col_name = "ma_period_" + str(self.period)

        df = ohlc_df.copy()

        # Intermediary Computations - Stage 1
        if self.type == MAType.ema:
            df[signal_col_name] = df["Adj Close"].ewm(
                span=self.period,
                min_periods=self.period
            ).mean()
        else:
            df[signal_col_name] = df["Adj Close"].rolling(
                window=self.period
            ).mean()

        return df, [signal_col_name]
