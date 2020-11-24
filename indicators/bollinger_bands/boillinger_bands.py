from typing import List, Tuple

import pandas as pd

from indicators.Indicator import Indicator


class BollingerBands(Indicator):

    ohlc_df: pd.DataFrame
    period: int

    def __init__(self, ohlc_df: pd.DataFrame, period: int = 14):
        self.ohlc_df = ohlc_df.__deepcopy__()
        self.period = period

    def get_signal(
            self
    ) -> Tuple[pd.DataFrame, List[str]]:

        """
        Calculates and returns the boilinger bands
        :return: Complete Dataframe and its corresponding 4 columns
        """

        self.ohlc_df["MA"] = self.ohlc_df['Adj Close'].rolling(self.period).mean()
        self.ohlc_df["BB_up"] = self.ohlc_df["MA"] + 2 * self.ohlc_df['Adj Close'].rolling(self.period).std(ddof=0)
        self.ohlc_df["BB_dn"] = self.ohlc_df["MA"] - 2 * self.ohlc_df['Adj Close'].rolling(self.period).std(ddof=0)
        self.ohlc_df["BB_width"] = self.ohlc_df["BB_up"] - self.ohlc_df["BB_dn"]

        del self.ohlc_df["MA"]

        return self.ohlc_df, ["BB_up", "BB_dn", "BB_width"]
