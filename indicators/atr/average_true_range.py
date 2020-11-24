from typing import List, Tuple

import pandas as pd

from indicators.Indicator import Indicator


class AverageTrueRange(Indicator):

    ohlc_df: pd.DataFrame
    period: int

    def __init__(self, ohlc_df: pd.DataFrame, period: int = 14):
        self.ohlc_df = ohlc_df.__deepcopy__()
        self.period = period

    def get_signal(
            self
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Calculates and returns the average true range
        :return: Complete Dataframe and its corresponding signal column
        """

        self.ohlc_df['H-L'] = abs(self.ohlc_df['High'] - self.ohlc_df['Low'])
        self.ohlc_df['H-PC'] = abs(self.ohlc_df['High'] - self.ohlc_df['Adj Close'].shift(1))
        self.ohlc_df['L-PC'] = abs(self.ohlc_df['Low'] - self.ohlc_df['Adj Close'].shift(1))

        # This could be an important param
        self.ohlc_df['TR'] = self.ohlc_df[['H-L', 'H-PC', 'L-PC']].max(axis=1, skipna=False)

        self.ohlc_df['ATR'] = self.ohlc_df['TR'].rolling(self.period).mean()

        self.ohlc_df2 = self.ohlc_df.drop(['H-L', 'H-PC', 'L-PC'], axis=1)

        return self.ohlc_df, ['ATR', 'TR']
