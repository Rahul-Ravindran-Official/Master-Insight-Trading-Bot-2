import unittest

from indicators.slope.point_slope import PointSlope
from market_data.ohlc_data import obtain_ohlc_data
import matplotlib.pyplot as plt
import pandas as pd

class PointSlopeTest(unittest.TestCase):
    def test_point_slope_constant_rising(self):
        dataset = pd.DataFrame([1, 2, 3, 4, 5], columns=['Adj Close'])
        df, col = PointSlope().get_signal(dataset)
        df[["Adj Close", col[0]]].plot()

        # All elements of a list are equal to 1
        self.assertEqual(df[col[0]].max(), 1)
        self.assertEqual(df[col[0]].max(), df[col[0]].min())

        plt.show()

    def test_point_slope_boxed(self):
        dataset = pd.DataFrame([1, 0, 1, 0, 1, 0, 1, 0], columns=['Adj Close'])
        df, col = PointSlope().get_signal(dataset)
        df[["Adj Close", col[0]]].plot()

        # All elements of a list are equal to 1
        middle_elements = df[col[0]][1:-1]
        self.assertEqual(middle_elements.max(), 0)
        self.assertEqual(middle_elements.max(), middle_elements.min())

        plt.show()

if __name__ == '__main__':
    unittest.main()
