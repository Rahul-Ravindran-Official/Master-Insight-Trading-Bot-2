import unittest

from indicators.ma.ma_type import MAType
from indicators.ma.moving_average import MovingAverage
from indicators.slope.point_slope import Slope
from market_data.ohlc_data import obtain_ohlc_data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from transformations.WaveletTransform import WaveletTransform


class PointSlopeTest(unittest.TestCase):
    def test_slope_constant_rising(self):
        dataset = pd.DataFrame([1, 2, 3, 4, 5], columns=['Adj Close'])
        df, col = Slope(dataset).get_signal()
        df[["Adj Close", col[0]]].plot()

        # All elements of a list are equal to 1
        self.assertEqual(df[col[0]].max(), 1)
        self.assertEqual(df[col[0]].max(), df[col[0]].min())

        plt.show()

    def test_slope_boxed(self):
        dataset = pd.DataFrame([1, 0, 1, 0, 1, 0, 1, 0], columns=['Adj Close'])
        df, col = Slope(dataset).get_signal()
        df[["Adj Close", col[0]]].plot()

        # All elements of a list are equal to 1
        middle_elements = df[col[0]][1:-1]
        self.assertEqual(middle_elements.max(), 0)
        self.assertEqual(middle_elements.max(), middle_elements.min())

        plt.show()

    def test_data_driven_slope(self):
        dataset = obtain_ohlc_data('MSFT')
        df, col = Slope(dataset).get_signal()
        df = df[:20]

        plt.scatter(df.index, df[col[0]], c='r')
        plt.scatter(df.index, df[col[1]], c='g')
        plt.plot(df.index, df['Adj Close'])
        df.plot()
        plt.show()

    def test_scatter(self):

        df = obtain_ohlc_data('MSFT')
        df, cols = Slope(df).get_signal()

        df = df[:10]

        # Plot results
        plt.scatter(df.index, df['local_minima'], c='r')
        # plt.scatter(df.index, df['local_maxima'], c='g')
        # plt.scatter(df.index, df[cols[0]], c='b')
        df['Adj Close'].plot()
        plt.show()


if __name__ == '__main__':
    unittest.main()
