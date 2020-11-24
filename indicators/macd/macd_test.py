import unittest

import numpy

from indicators.macd.macd import Macd
from market_data.ohlc_data import obtain_ohlc_data
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt

class macdTest(unittest.TestCase):

    def test_macd_count(self):
        dataset = obtain_ohlc_data('MSFT')
        macd_included_df, sig_cols = Macd(12, 26, 9).get_signal(dataset)
        self.assertEqual(len(macd_included_df.get(sig_cols).count(axis=0)), 2)

    def test_macd_columns(self):
        dataset = obtain_ohlc_data('MSFT')
        macd_included_df, sig_cols = Macd(12, 26, 9).get_signal(dataset)
        self.assertEqual(
            macd_included_df.get(sig_cols).count(axis=0).index.tolist(),
            ['MACD', 'Signal']
        )

    def test_datatypes(self):
        dataset = obtain_ohlc_data('MSFT')
        macd_included_df, sig_cols = Macd(12, 26, 9).get_signal(dataset)
        self.assertEqual(type(macd_included_df['Signal'][0]), numpy.float64)
        self.assertEqual(type(macd_included_df['MACD'][0]), numpy.float64)

    def test_macd_validity(self):
        pass
        # dataset_raw = [
        #     459.99, 448.85, 446.06, 450.81, 442.8, 448.97,
        #     444.57, 441.4, 430.47, 420.05, 431.14, 425.66,
        #     430.58, 431.72, 437.87, 428.43, 428.35
        # ]
        # dataset = pd.DataFrame(dataset_raw, columns=['Adj Close'])
        # macd_included_df, sig_cols = Macd(5, 7, 9).get_signal(dataset)
        # print(1==1)

    def test_macd_visualization(self):
        dataset = obtain_ohlc_data('MSFT')
        macd_included_df, sig_cols = Macd(12, 26, 9).get_signal(dataset)
        # macd_included_df = (macd_included_df - macd_included_df.min()) / (macd_included_df.max() - macd_included_df.min())
        macd_included_df[['MACD', 'Signal']][:100].plot()
        plt.show()
        # macd_included_df[['Adj Close']][:100].plot()
        # plt.show()
        print(macd_included_df)


if __name__ == '__main__':
    unittest.main()
