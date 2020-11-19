from typing import Dict, List, Tuple
import pandas as pd
from shared.Strategy import Strategy


class SignalsProcurer:
    ohlc_data: pd.DataFrame
    strategies: Dict[Strategy, float]
    strategy_signal_columns: List[str]
    strategy_weightage: List[float]

    def __init__(
            self,
            ohlc_data: pd.DataFrame,
            strategies: Dict[Strategy, float]
    ):
        self.ohlc_data = ohlc_data
        self.strategies = strategies
        self.strategy_signal_columns = []
        self.strategy_weightage = []

    def procure(self) -> Tuple[pd.DataFrame, List[str], List[float]]:
        for strategy in self.strategies:
            self.ohlc_data, bbhss_col = strategy.\
                get_bbhss_signal(
                    self.ohlc_data
                )
            self.strategy_signal_columns.append(bbhss_col)
            self.strategy_weightage.append(self.strategies[strategy])

        return self.ohlc_data, self.strategy_signal_columns, self.strategy_weightage
