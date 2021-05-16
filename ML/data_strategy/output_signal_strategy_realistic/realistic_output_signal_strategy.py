import pandas as pd
import numpy as np


class RealisticOutputSignalStrategy:

    def __init__(
            self,
            ohlc_data: pd.DataFrame,
            commission: float,
            relative_min_trade_value: float
    ):
        self.ohlc_data = ohlc_data
        self.commission = commission
        self.relative_min_trade_value = relative_min_trade_value

    def process(self) -> np.array:
        pass
