import unittest
from typing import Dict

from backtesting.signal_processor.signal_manager import SignalManager
from ma_double_cross_over import MADoubleCrossOver
from market_data.ohlc_data import obtain_ohlc_data
from shared.Strategy import Strategy
import pandas as pd

class SignalTester(unittest.TestCase):
    def test_basic_sig_merge(self):
        ohlc_data: pd.DataFrame = obtain_ohlc_data('AAPL')
        strategies: Dict[Strategy, float] = {
            MADoubleCrossOver(1, 5, 13): 0.5,
            MADoubleCrossOver(2, 50, 200): 0.5,
        }

        ohlc_data = SignalManager(
            ohlc_data,
            strategies
        ).get_master_signal()

        self.assertEqual(len(ohlc_data.get('master_sig')), len(ohlc_data.get('open')))


if __name__ == '__main__':
    unittest.main()
