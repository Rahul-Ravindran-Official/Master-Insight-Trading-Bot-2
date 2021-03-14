import unittest

import numpy as np

from indicators.ma.ma_type import MAType
from indicators.ma.moving_average import MovingAverage
from market_data.ohlc_data import obtain_ohlc_data
import pandas as pd
import matplotlib.pyplot as plt



class MovingAverageTest(unittest.TestCase):

    def test_ma_count(self):
        dataset = obtain_ohlc_data('MSFT')
        ma_ohlc, sig_cols = MovingAverage(12, MAType.ema).get_signal(dataset)
        self.assertEqual(len(sig_cols), 1)

    def test_ma_columns(self):
        dataset = obtain_ohlc_data('MSFT')
        period: int = 12
        ma_ohlc, sig_cols = MovingAverage(period, MAType.ema).get_signal(dataset)
        self.assertIn(
            sig_cols[0],
            ma_ohlc.count(axis=0).index.tolist()
        )

    def test_datatypes(self):
        dataset = obtain_ohlc_data('MSFT')
        period: int = 12
        ma_ohlc, sig_cols = MovingAverage(period, MAType.ema).get_signal(dataset)
        self.assertEqual(type(ma_ohlc.get(sig_cols)['ma_period_'+str(period)][0]), np.float64)

    def test_sma_computation(self):
        dataset_input = [290, 260, 288, 300, 310]
        dataset_output = [279.33, 282.67, 299.33]
        period: int = 3
        dataset = pd.DataFrame({'Adj Close': dataset_input})
        ma_ohlc, sig_cols = MovingAverage(period, MAType.sma).get_signal(dataset)
        ma_ohlc.dropna(inplace=True)
        self.assertListEqual(list(np.around(ma_ohlc.get(sig_cols).ma_period_3, 2)), dataset_output)

    def test_ma_visualization(self):
        dataset = obtain_ohlc_data('MSFT')[:300]
        period: int = 7
        ma_ohlc, sig_cols = MovingAverage(dataset, period, MAType.ema).get_signal()



        plt.plot(ma_ohlc.index, ma_ohlc["Adj Close"], label="Adj Close")
        plt.plot(ma_ohlc.index, ma_ohlc["ma_period_7"], label="MA")
        plt.show()

    def test_ema_computation(self):
        # todo
        pass


if __name__ == '__main__':
    unittest.main()
