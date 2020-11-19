from typing import List
import pandas as pd
import numpy as np

class SignalCombiner:
    ohlc_data: pd.DataFrame
    strategy_signal_columns: List[str]
    strategy_weightage: List[float]
    should_remove_signal_columns: bool

    def __init__(
            self,
            ohlc_data: pd.DataFrame,
            strategy_signal_columns: List[str],
            strategy_weightage: List[float],
            should_remove_signal_columns: bool = True
    ):
        self.ohlc_data = ohlc_data
        self.strategy_signal_columns = strategy_signal_columns
        self.strategy_weightage = strategy_weightage
        self.should_remove_signal_columns = should_remove_signal_columns

    def combine(self) -> pd.DataFrame:
        """
        B S B | 0.5         |0.5*B + 0.3*S + 0.2*B|
        B S B | 0.3     =>  |0.5*B + 0.3*S + 0.2*B|
        B B S | 0.2         |0.5*B + 0.3*B + 0.2*S|
        :return: combined signal
        """

        strategy_weightage_np_array = pd.DataFrame(
            self.strategy_weightage,
            columns=['weightage']
        ).to_numpy()

        strategy_signal_columns_np_array = self.ohlc_data[self.strategy_signal_columns].to_numpy()

        self.ohlc_data['master_signal'] = np.dot(
            strategy_signal_columns_np_array,
            strategy_weightage_np_array
        )

        self.remove_signal_columns()

        return self.ohlc_data

    def remove_signal_columns(self):
        if self.remove_signal_columns:
            self.ohlc_data.drop(columns=self.strategy_signal_columns, inplace=True)
