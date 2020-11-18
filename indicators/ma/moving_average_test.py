import unittest

import numpy as np

from indicators.ma.ma_type import MAType
from indicators.ma.moving_average import MovingAverage
from market_data.ohlc_data import obtain_ohlc_data
import pandas as pd


class MovingAverageTest(unittest.TestCase):

    def test_ma_count(self):
        dataset = obtain_ohlc_data('MSFT')
        ma_indicator = MovingAverage(12, MAType.ema).get_signal(dataset)
        self.assertEqual(len(ma_indicator.count(axis=0)), 1)

    def test_macd_columns(self):
        dataset = obtain_ohlc_data('MSFT')
        period: int = 12
        ma_indicator = MovingAverage(period, MAType.ema).get_signal(dataset)
        self.assertEqual(
            ma_indicator.count(axis=0).index.tolist(),
            ['ma_period_'+str(period)]
        )

    def test_datatypes(self):
        dataset = obtain_ohlc_data('MSFT')
        period: int = 12
        ma_indicator = MovingAverage(period, MAType.ema).get_signal(dataset)
        self.assertEqual(type(ma_indicator['ma_period_'+str(period)][0]), np.float64)

    def test_sma_computation(self):
        dataset_input = [290, 260, 288, 300, 310]
        dataset_output = [279.33, 282.67, 299.33]
        period: int = 3
        dataset = pd.DataFrame({'Adj Close': dataset_input})
        ma_indicator = MovingAverage(period, MAType.sma).get_signal(dataset)
        self.assertListEqual(list(np.around(ma_indicator.ma_period_3, 2)), dataset_output)

    def test_ema_computation(self):
        # todo
        pass


if __name__ == '__main__':
    unittest.main()
