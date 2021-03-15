from backtesting.signal_processor.signal_manager import SignalManager
from bollinger_bands_simple_breakout import BollingerBandsSimpleBreakout
from ma_double_cross_over import MADoubleCrossOver
import pandas as pd

class InputSignalMatrix:
    """

    """

    def __init__(self, dataset: pd.DataFrame):
        self.ohlc_data = dataset
        self.signal_manager = SignalManager(
            self.ohlc_data,
            {
                MADoubleCrossOver(
                    magic_no=1,
                    period_fast=5,
                    period_slow=10
                ): 0.33,
                MADoubleCrossOver(
                    magic_no=3,
                    period_fast=50,
                    period_slow=200
                ): 0.33,
                BollingerBandsSimpleBreakout(
                    magic_no=2,
                    period=14
                ): 0.34
            }
        )

    def get_input_matrix(self) -> pd.DataFrame:
        return self.signal_manager.get_raw_signals()


if __name__ == "__main__":
    pass
