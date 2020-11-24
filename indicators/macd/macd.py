from typing import List, Tuple

import pandas as pd

from indicators.Indicator import Indicator


class Macd(Indicator):
    """
    Calcluates The MACD and it's infamous signal
    """

    ohlc_df: pd.DataFrame
    period_fast: int
    period_slow: int
    signal_period: int


    def __init__(
            self,
            ohlc_df: pd.DataFrame,
            period_fast: int = 12,
            period_slow: int = 26,
            signal_period: int = 9,
    ):
        """
        :param ohlc_df: Dataframe containing security data
        :param period_fast: Fast MA line
        :param period_slow: Slow MA line
        :param signal_period: MA of signal
        """
        self.ohlc_df = ohlc_df.__deepcopy__()
        self.period_fast = period_fast
        self.period_slow = period_slow
        self.signal_period = signal_period

    def get_signal(
            self
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Calculates and returns MACD & Signal
        :return: Dataframe with columns "MACD" and "Signal"
        """

        # Intermediary Computations - Stage 1
        self.ohlc_df["MA_Fast"] = self.ohlc_df["Adj Close"].ewm(
            span=self.period_fast,
            min_periods=self.period_fast
        ).mean()

        self.ohlc_df["MA_Slow"] = self.ohlc_df["Adj Close"].ewm(
            span=self.period_slow,
            min_periods=self.period_slow
        ).mean()

        # Intermediary Computations - Stage 2
        self.ohlc_df["MACD"] = self.ohlc_df["MA_Fast"] - self.ohlc_df["MA_Slow"]
        self.ohlc_df["Signal"] = self.ohlc_df["MACD"].ewm(
            span=self.signal_period,
            min_periods=self.signal_period).mean()

        del self.ohlc_df["MA_Slow"]
        del self.ohlc_df["MA_Fast"]

        return self.ohlc_df, ["MACD", "Signal"]
