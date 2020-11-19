import unittest

import numpy

from indicators.macd.macd import Macd
from market_data.ohlc_data import obtain_ohlc_data

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

if __name__ == '__main__':
    unittest.main()
