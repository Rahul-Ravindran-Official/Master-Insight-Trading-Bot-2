from typing import List, Tuple

import pandas as pd

from indicators.Indicator import Indicator
import numpy as np

class PointSlope(Indicator):

    def get_signal(
            self,
            ohlc_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Calculates and returns the point slope of every
        "Adj Close" price point by taking a negligible
        delta on either side

        :param ohlc_df: A yfinance OHLC Dataframe
        :return: Complete Dataframe and "Point_Slope"
        """

        df = ohlc_df.copy()

        df['Point_Slope'] = np.gradient(np.array(df['Adj Close'], dtype=np.float))

        return df, ['Point_Slope']
