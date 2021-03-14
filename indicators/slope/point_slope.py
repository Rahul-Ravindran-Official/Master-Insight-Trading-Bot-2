from typing import List, Tuple

import pandas as pd
from indicators.Indicator import Indicator
import numpy as np


class Slope(Indicator):

    ohlc_df: pd.DataFrame

    def __init__(self, ohlc_df: pd.DataFrame):
        """
        :param ohlc_df: Dataframe containing security data
        """
        self.ohlc_df = ohlc_df.__deepcopy__()

    def get_signal(
            self
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Calculates and returns slope of every
        "Adj Close" price point

        :return: Complete Dataframe and "Point_Slope"
        """

        self.ohlc_df['Point_Slope'] = np.gradient(np.array(self.ohlc_df['Adj Close'], dtype=np.float))

        # Find local peaks
        self.ohlc_df['local_minima'] = (self.ohlc_df['Adj Close'].shift(1) > self.ohlc_df['Adj Close']) & (self.ohlc_df['Adj Close'].shift(-1) > self.ohlc_df['Adj Close'])
        self.ohlc_df['local_maxima'] = (self.ohlc_df['Adj Close'].shift(1) < self.ohlc_df['Adj Close']) & (self.ohlc_df['Adj Close'].shift(-1) < self.ohlc_df['Adj Close'])

        self.ohlc_df['binary_min'] = (self.ohlc_df['Adj Close'].shift(1) > self.ohlc_df['Adj Close']) & (self.ohlc_df['Adj Close'].shift(-1) > self.ohlc_df['Adj Close'])
        self.ohlc_df['binary_max'] = (self.ohlc_df['Adj Close'].shift(1) < self.ohlc_df['Adj Close']) & (self.ohlc_df['Adj Close'].shift(-1) < self.ohlc_df['Adj Close'])

        self.ohlc_df['merged_min_max'] = np.select([self.ohlc_df.local_maxima >= float('-inf'), self.ohlc_df.local_minima >= float('-inf')], [self.ohlc_df.local_maxima, self.ohlc_df.local_minima])
        self.ohlc_df['merged_min_max'] = self.ohlc_df['merged_min_max'].replace({0: np.nan})

        del_range = []
        for i in range(self.ohlc_df.shape[0]):
            zero_slope = self.ohlc_df[['merged_min_max']].iloc[i]
            if not np.isnan(zero_slope[0]):
                del_range.append(i)
            elif len(del_range) > 1:
                for idx in del_range:

                    self.ohlc_df.loc[self.ohlc_df.index[idx], 'merged_min_max'] = np.nan
                    self.ohlc_df.loc[self.ohlc_df.index[idx], 'binary_min'] = False
                    self.ohlc_df.loc[self.ohlc_df.index[idx], 'binary_max'] = False

                del_range = []
            else:
                del_range = []

        return self.ohlc_df, ['merged_min_max', 'local_maxima', 'local_minima']
