import unittest

import numpy

from indicators.macd.macd import Macd
from market_data.ohlc_data import obtain_ohlc_data

class macdTest(unittest.TestCase):

    def test_macd_count(self):
        dataset = obtain_ohlc_data('MSFT')
        macd_indicator = Macd(12, 26, 9).get_signal(dataset)
        self.assertEqual(len(macd_indicator.count(axis=0)), 2)

    def test_macd_columns(self):
        dataset = obtain_ohlc_data('MSFT')
        macd_indicator = Macd(12, 26, 9).get_signal(dataset)
        self.assertEqual(
            macd_indicator.count(axis=0).index.tolist(),
            ['MACD', 'Signal']
        )

    def test_equal_rows(self):
        dataset = obtain_ohlc_data('MSFT')
        macd_indicator = Macd(12, 26, 9).get_signal(dataset)
        macd_count, signal_count = macd_indicator.count(axis=0).to_list()
        self.assertEqual(macd_count, signal_count)

    def test_datatypes(self):
        dataset = obtain_ohlc_data('MSFT')
        macd_indicator = Macd(12, 26, 9).get_signal(dataset)
        self.assertEqual(type(macd_indicator['Signal'][0]), numpy.float64)
        self.assertEqual(type(macd_indicator['MACD'][0]), numpy.float64)

if __name__ == '__main__':
    unittest.main()
