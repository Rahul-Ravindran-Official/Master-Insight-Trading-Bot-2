from typing import Dict, List, Tuple

from backtesting.signal_processor.signal_combiner import SignalCombiner
from backtesting.signal_processor.signals_procurer import SignalsProcurer
from shared.Strategy import Strategy
import pandas as pd


class SignalManager:
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

    def get_master_signal(self):
        self.ohlc_data, self.strategy_signal_columns, self.strategy_weightage =\
            self.obtain_bbhss_signals()
        get_df_with_master_sig = self.merge_bbhss_signals()
        return get_df_with_master_sig

    def obtain_bbhss_signals(self) -> Tuple[pd.DataFrame, List[str], List[float]]:
        return SignalsProcurer(self.ohlc_data, self.strategies)\
            .procure()

    def merge_bbhss_signals(self) -> pd.DataFrame:
        return SignalCombiner(
            self.ohlc_data,
            self.strategy_signal_columns,
            self.strategy_weightage
        ).combine()
