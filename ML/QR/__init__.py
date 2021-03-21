import unittest
from typing import List

from ML.QR.multi_signal_predictor import MultiSignalPredictor
from ML.data_strategy.i_o_data_provider import IOProvider
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from scipy.signal import lfilter
import pandas as pd
from market_data.ohlc_data import obtain_ohlc_data


class QRH_Strategy_Tester(unittest.TestCase):

    def init(
            self, ticker: str,
            prediction_cutoff: float,
            interested_data: str = "train_binary_signal",
            interested_data_smoothening: int = 1,
            show_graphs=3,
            training_signal="get_buy_signals"
            # training_signal="get_gradient_signals"
            # training_signal="get_pred_profit_signals"
    ):

        iop = IOProvider(ticker)

        input_matrix = iop.obtain_input_matrix()
        output_vector, self.start_end_set = iop.obtain_output_vector(training_signal)

        pd.DataFrame(input_matrix).to_csv(
            r'/Users/rahul/Main/CloudStation/Spizen/spizen-forex/master-insight-trading-bot-2/ML/data_strategy/output_signal_strategy/input.csv',
            index=False
        )

        pd.DataFrame(output_vector).to_csv(
            r'/Users/rahul/Main/CloudStation/Spizen/spizen-forex/master-insight-trading-bot-2/ML/data_strategy/output_signal_strategy/target.csv',
            index=False
        )

        msp = MultiSignalPredictor(
            input_matrix,
            output_vector,
            0.8,
            prediction_cutoff
        )

        msp.compute_magic_numbers()

        v_dict = {
            "train_probability": [msp.prediction_vector, msp.output_signals_train],
            "train_binary_signal": [msp.prediction_vector_signal, msp.output_signals_train],
            "test_probability": [msp.prediction_vector_test, msp.output_signals_test],
            "test_binary_signal": [msp.prediction_vector_test_signal, msp.output_signals_test],
        }

        # Normalised Prediction Data
        prediction_vector_normalised = self.normalize_0_1_filter(v_dict[interested_data][0])

        # self.normalize_every_range(v_dict[interested_data][0])

        # Smoothening
        smoothened = self.smoothen_filter(prediction_vector_normalised, interested_data_smoothening)

        # Visualization
        self.visualize(smoothened, self.normalize_0_1_filter(v_dict[interested_data][1]), show_graphs)

    def visualize(self, line_1, line_2, show_graphs:int = 3):

        if show_graphs == 1 or show_graphs == 3:
            plt.plot(self.smoothen_filter(line_1[0:], 10), c="b")

        if show_graphs == 2 or show_graphs == 3:
            plt.plot(line_2[0:], c="r")

        # plt.plot(minmax_scale(obtain_ohlc_data('AAPL')["Close"].to_numpy()[0:1000],feature_range=(-1, 1),axis=0,copy=False))

        plt.show()

    def smoothen_filter(self, data_to_smooth, smooth_factor):
        data = data_to_smooth
        n = smooth_factor
        b = [1.0 / n] * n
        a = 1
        yy = lfilter(b, a, data)
        yy[0] = yy[1]
        return yy

    def normalize_0_1_filter(self, data):
        return minmax_scale(
            data,
            feature_range=(0, 1),
            axis=0,
            copy=True
        )

    def normalize_every_range(self, data):
        i = 0
        for s in self.start_end_set:
            print(i)
            i+=1
            i0 = s[0]
            i1 = s[1]
            try:
                d = data[i0:i1]
                dfil = self.smoothen_filter(d, 10)
                data[i0:i1] = minmax_scale(dfil,feature_range=(-1, 1),axis=0,copy=True)
            except:
                print("Did not normalize tuple: " + str(s))

    def test_train_probability(self):
        self.init(
            ticker="AAPL",
            prediction_cutoff=0.50,
            interested_data="train_probability",
            interested_data_smoothening=1,
            show_graphs=3
        )

    def test_train_binary_signal(self):
        self.init(
            ticker="AAPL",
            prediction_cutoff=0.90,
            interested_data="train_binary_signal",
            interested_data_smoothening=1,
            show_graphs=3
        )

    def test_test_probability(self):
        self.init(
            ticker="AAPL",
            prediction_cutoff=0.50,
            interested_data="test_probability",
            interested_data_smoothening=1,
            show_graphs=3
        )

    def test_test_binary_signal(self):
        self.init(
            ticker="AAPL",
            prediction_cutoff=0.90,
            interested_data="test_binary_signal",
            interested_data_smoothening=1,
            show_graphs=3
        )

if __name__ == "__main__":
    qst = QRH_Strategy_Tester()
    qst.test_train_probability()

# if __name__ == "__main__":
#
#     iop = IOProvider("AAPL")
#     input_matrix = iop.obtain_input_matrix()
#     output_vector = iop.obtain_output_vector()
#
#     msp = MultiSignalPredictor(
#         input_matrix,
#         output_vector,
#         0.8,
#         0.40
#     )
#
#     msp.compute_magic_numbers()
#
#     # msp_2 = MultiSignalPredictor(
#     #     np.insert(input_matrix, 3, values=msp.prediction_vector, axis=1),
#     #     output_vector,
#     #     0.8
#     # )
#     #
#     # msp_2.compute_magic_numbers()
#
#
#
#     # prediction_vector_normalised = minmax_scale(msp.prediction_vector, feature_range=(0, 1), axis=0, copy=True)
#     prediction_vector_normalised = minmax_scale(msp.prediction_vector_signal, feature_range=(0, 1), axis=0, copy=True)
#
#     # Smoothening
#
#     data = prediction_vector_normalised
#     n = 1
#     b = [1.0 / n] * n
#     a = 1
#     yy = lfilter(b, a, data)
#     yy[0] = yy[1]
#
#     # Smoothening
#
#
#
#     plt.plot(yy[:], c="b")
#     plt.plot(msp.output_signals_train[:], c="r")
#
#     plt.show()
#
#     i = 0


