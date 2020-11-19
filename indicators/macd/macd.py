from typing import List, Tuple

import pandas as pd

from indicators.Indicator import Indicator


class Macd(Indicator):
    period_fast: int
    period_slow: int
    signal_period: int

    def __init__(
            self,
            period_fast: int = 12,
            period_slow: int = 26,
            signal_period: int = 9
    ):
        self.period_fast = period_fast
        self.period_slow = period_slow
        self.signal_period = signal_period

    def get_signal(
            self,
            ohlc_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Calculates and returns MACD & Signal

        :param ohlc_df: A yfinance OHLC Dataframe
        :param period_fast: Fast MA line
        :param period_slow: Slow MA line
        :param signal_period: MA of signal
        :return: Dataframe with columns "MACD" and "Signal"
        """
        df = ohlc_df.copy()

        # Intermediary Computations - Stage 1
        df["MA_Fast"] = df["Adj Close"].ewm(
            span=self.period_fast,
            min_periods=self.period_fast
        ).mean()
        df["MA_Slow"] = df["Adj Close"].ewm(
            span=self.period_slow,
            min_periods=self.period_slow
        ).mean()

        # Intermediary Computations - Stage 2
        df["MACD"] = df["MA_Fast"] - df["MA_Slow"]
        df["Signal"] = df["MACD"].ewm(
            span=self.signal_period,
            min_periods=self.signal_period).mean()

        return df, ["MACD", "Signal"]
