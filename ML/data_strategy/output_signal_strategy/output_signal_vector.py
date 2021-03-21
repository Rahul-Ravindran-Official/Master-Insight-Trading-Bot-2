import unittest
from typing import List, Tuple

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
        self.start_end_set = []

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

    def signal_mux(self, signal: str):
        if signal == "get_buy_signals":
            return self.get_buy_signals()
        elif signal == "get_sell_signals":
            return self.get_sell_signals()
        elif signal == "get_gradient_signals":
            return self.get_gradient_signals()
        elif signal == "get_pred_profit_signals":
            return self.get_pred_profit_signals()

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

    def get_gradient_signals(self) -> np.array:
        output_signal = np.zeros((1, self.count))

        # Add In Maxima, Minima Signals
        ## Add in Maxima Signal
        for i in self.sig_peaks:
            output_signal[0][i] = -1

        ## Add in Minima Signal
        for i in self.sig_throughs:
            output_signal[0][i] = 1

        # Add in Gradient Signals
        self.get_pred_profit_signals()
        return self.generate_intermediate_signals(output_signal)

    def get_pred_profit_signals(self) -> np.array:
        output_signal = np.zeros((1, self.count))

        # Add In Maxima, Minima Signals
        ## Add in Maxima Signal
        for i in self.sig_peaks:
            output_signal[0][i] = -1

        ## Add in Minima Signal
        for i in self.sig_throughs:
            output_signal[0][i] = 1

        # Add in Pred Profits Signals
        self.start_end_set = self.generate_min_max_pairs(output_signal)


        for s in self.start_end_set:
            start_point = s[0]
            end_point = s[1]
            price_point_at_sp = self.ohlc_data["Adj Close"][end_point]

            for i in range(start_point, end_point):
                output_signal[0][i] = price_point_at_sp - self.ohlc_data["Adj Close"][i]

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

    def generate_min_max_pairs(self, input: np.array):
        # A = np.array([0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1])

        start_end_set: List[Tuple[int, int, int]] = []

        sign_arr = []
        idx_arr = []

        for i in range(len(input[0])):

            sign = input[0][i]

            if sign == 1 or sign == -1:

                sign_arr.append(sign)
                idx_arr.append(i)

                if len(sign_arr) == 2:
                    start_end_set.append(
                        (idx_arr[0], idx_arr[1], sign_arr[0]))
                    sign_arr.pop(0)
                    idx_arr.pop(0)
        return start_end_set

    def generate_intermediate_signals(self, input: np.array):

        self.start_end_set = self.generate_min_max_pairs(input)

        a = np.array(input, dtype=np.object)

        # Apply Ranges
        for s in self.start_end_set:
            i0 = s[0]
            i1 = s[1]
            sign = s[2]

            element_count = i1 - i0

            if sign == 1:
                a[0][i0+1:i1] = np.arange(1, -1, - (2 / element_count))[1:]
            else:
                a[0][i0 + 1:i1] = np.arange(-1, 1, (2 / element_count))[1:]

        return np.array(a, dtype=np.float)

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
