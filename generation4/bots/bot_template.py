from typing import Dict
import numpy as np
import pandas as pd

class TradingBot:

    _tr_x: pd.DataFrame
    _tr_y: pd.DataFrame
    _va_x: pd.DataFrame
    _metadata: Dict

    def __init__(
            self,
            tr_x: pd.DataFrame,
            tr_y: pd.DataFrame,
            va_x: pd.DataFrame,
            metadata: Dict
    ):
        self._tr_x = tr_x
        self._tr_y = tr_y
        self._va_x = va_x
        self._metadata = metadata

    def predict(self) -> np.array:
        pass
