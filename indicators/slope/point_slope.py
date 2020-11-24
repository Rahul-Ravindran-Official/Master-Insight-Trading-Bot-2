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

        return self.ohlc_df, ['Point_Slope']
