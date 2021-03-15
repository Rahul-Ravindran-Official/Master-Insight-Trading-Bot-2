import unittest
import matplotlib.pyplot as plt
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
import scipy.signal as sig
import numpy as np
from scipy.signal import lfilter
import pandas as pd

from market_data.ohlc_data import obtain_ohlc_data

class OutputSignalVector:

    ohlc_data: pd.DataFrame
    price_points: np.array
    count: int
    smooth_intensity: int
    threshold: float
    profit: float
    sig_peaks: np.array
    sig_throughs: np.array

    def __init__(self, ohlc_data, count: int = 300, smooth_intensity: int = 2, threshold: float=1.5):

        if count == -1:
            self.count = ohlc_data.shape[0]
        else:
            self.count = count

        self.ohlc_data = ohlc_data
        self.price_points = self.ohlc_data["Adj Close"].to_numpy()
        self.smooth_intensity = smooth_intensity
        self.threshold = threshold
        self.compute_signals()

    def compute_signals(self):

        x = self.smooth_algo_1(self.price_points, self.count, self.smooth_intensity)

        # Finding Local Maxima and Minima
        peaks, _ = find_peaks(x, height=0)
        throughs, _ = find_peaks(-x)

        # Remove Nearby Points

        peaks_and_throughs = np.concatenate((peaks, throughs))
        peaks_and_throughs.sort(kind='mergesort')

        cleansed = []
        for n in peaks_and_throughs:
            if not cleansed or abs(x[n] - x[cleansed[-1]]) >= self.threshold:
                cleansed.append(n)

        cleansed = np.array(cleansed)

        peaks = np.array(list(filter(lambda x: x in cleansed, peaks)))
        throughs = np.array(list(filter(lambda x: x in cleansed, throughs)))

        # If this point is a peak/through and next is the same then remove current point
        cleansed_consecutive_peaks = []
        cleansed_consecutive_throughs = []
        idx = 0
        for n in cleansed:
            try:
                if n in peaks and cleansed[idx + 1] not in peaks:
                    cleansed_consecutive_peaks.append(n)

                if n in throughs and cleansed[idx + 1] not in throughs:
                    cleansed_consecutive_throughs.append(n)

                idx += 1
            except:
                pass

        peaks = cleansed_consecutive_peaks
        throughs = cleansed_consecutive_throughs

        # Calculating Profit
        self.profit = self.profit_function(peaks, throughs, x)

        self.sig_peaks = peaks
        self.sig_throughs = throughs

    def get_buy_signals(self) -> np.array:
        output_signal = np.zeros((1, self.count))
        for i in self.sig_throughs:
            output_signal[0][i] = 1
        return output_signal

    def get_sell_signals(self) -> np.array:
        output_signal = np.zeros((1, self.count))
        for i in self.sig_peaks:
            output_signal[0][i] = 1
        return output_signal

    def smooth_algo_1(self, data, cnt: int = 200, smooth_intensity: int = 3):
        data = data[:cnt]
        n = smooth_intensity  # the larger n is, the smoother curve will be
        b = [1.0 / n] * n
        a = 1
        yy = lfilter(b, a, data)
        yy[0] = yy[1]
        return yy

    def profit_function(self, peaks, throughs, x):
        profit = 0

        max_min_range = max(x) - min(x)

        peaks_and_throughs = np.concatenate((peaks, throughs))
        for idx in range(len(peaks_and_throughs)-1):
            if peaks_and_throughs[idx] in peaks:
                # Sell Order
                p = x[peaks_and_throughs[idx]] - x[peaks_and_throughs[idx+1]]
                print(p)
                profit += (p / max_min_range) * 100
            else:
                # Buy Order
                p = x[peaks_and_throughs[idx + 1]] - x[peaks_and_throughs[idx]]
                print(p)
                profit += (p / max_min_range) * 100

        return profit

class PlaygroundOutputSignalMatrix(unittest.TestCase):

    def smooth_algo_1(self, data, cnt: int = 200, smooth_intensity: int = 3):
        data = data[:cnt]
        n = smooth_intensity  # the larger n is, the smoother curve will be
        b = [1.0 / n] * n
        a = 1
        yy = lfilter(b, a, data)[n - 1:]
        return yy

    def get_data(self):
        return obtain_ohlc_data('MSFT')["Adj Close"].to_numpy()

    def profit_function(self, peaks, throughs, x):
        profit = 0

        max_min_range = max(x) - min(x)

        peaks_and_throughs = np.concatenate((peaks, throughs))
        for idx in range(len(peaks_and_throughs)-1):
            if peaks_and_throughs[idx] in peaks:
                # Sell Order
                p = x[peaks_and_throughs[idx]] - x[peaks_and_throughs[idx+1]]
                print(p)
                profit += (p / max_min_range) * 100
            else:
                # Buy Order
                p = x[peaks_and_throughs[idx + 1]] - x[peaks_and_throughs[idx]]
                print(p)
                profit += (p / max_min_range) * 100

        return profit

    def test_approach_1_awesome(self):

        x = self.smooth_algo_1(self.get_data(), 400, 2)

        # Finding Local Maxima and Minima
        peaks, _ = find_peaks(x, height=0)
        throughs, _ = find_peaks(-x)

        # Remove Nearby Points
        treshold = 1.5

        peaks_and_throughs = np.concatenate((peaks, throughs))
        peaks_and_throughs.sort(kind='mergesort')

        cleansed = []
        for n in peaks_and_throughs:
            if not cleansed or abs(x[n] - x[cleansed[-1]]) >= treshold:
                cleansed.append(n)

        cleansed = np.array(cleansed)

        peaks = np.array(list(filter(lambda x: x in cleansed, peaks)))
        throughs = np.array(list(filter(lambda x: x in cleansed, throughs)))


        # If this point is a peak/through and next is the same then remove current point
        cleansed_consecutive_peaks = []
        cleansed_consecutive_throughs = []
        idx = 0
        for n in cleansed:
            try:
                if n in peaks and cleansed[idx+1] not in peaks:
                    cleansed_consecutive_peaks.append(n)

                if n in throughs and cleansed[idx+1] not in throughs:
                    cleansed_consecutive_throughs.append(n)

                idx += 1
            except:
                pass


        peaks = cleansed_consecutive_peaks
        throughs = cleansed_consecutive_throughs

        # Calculating Profit
        print("Profit Function: " + str(self.profit_function(peaks, throughs, x)))

        # Plotting
        plt.plot(x)
        plt.plot(peaks, x[peaks], "v")
        plt.plot(throughs, x[throughs], "^")

        plt.show()


    def test_approach_2(self):

        org = self.get_data()[:200]
        x = self.smooth_algo_1(self.get_data(), 200, 3)

        peaks, _ = find_peaks(x, distance=5)
        # difference between peaks is >= 150
        print(np.diff(peaks))
        # prints [186 180 177 171 177 169 167 164 158 162 172]

        plt.plot(x)
        plt.plot(peaks, x[peaks], "x")
        plt.plot(org, linewidth=1, linestyle="dashdot", c="g")
        plt.show()

    def test_approach_3(self):
        x = electrocardiogram()[200:300]
        peaks, _ = find_peaks(x)
        troughs, _ = find_peaks(-x)
        plt.plot(x)
        plt.plot(peaks, x[peaks], '^', c="r")
        plt.plot(troughs, x[troughs], 'v', c="g")
        plt.show()

    def test_approach_4(self):
        cnt = 200
        data = self.get_data()[:cnt]
        x = np.arange(1, cnt+1, 1)

        n = 4  # the larger n is, the smoother curve will be
        b = [1.0 / n] * n
        a = 1
        yy = lfilter(b, a, data)[n-1:]
        plt.plot(x, data, linewidth=1, linestyle="dashdot", c="r")  # smooth by filter
        plt.plot(x[n-1:], yy, linewidth=2, linestyle="-", c="b")  # smooth by filter
        plt.show()
