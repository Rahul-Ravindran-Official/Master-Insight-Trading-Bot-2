from backtesting.signal_processor.signal_manager import SignalManager
from bollinger_bands_simple_breakout import BollingerBandsSimpleBreakout
from ma_double_cross_over import MADoubleCrossOver
from market_data.ohlc_data import obtain_ohlc_data


class GetSignalMatrix:
    """

    """

    def __init__(self):
        self.ohlc_data = obtain_ohlc_data("AAPL")
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

        self.ohlc_data = self.signal_manager.get_raw_signals()





if __name__ == "__main__":
    a = GetSignalMatrix()
