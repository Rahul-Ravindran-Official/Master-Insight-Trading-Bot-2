from ML.data_strategy.input_signal_strategy.input_signal_matrix import \
    InputSignalMatrix
from ML.data_strategy.output_signal_strategy.output_signal_vector import \
    OutputSignalVector
from market_data.ohlc_data import obtain_ohlc_data
import pandas as pd
import numpy as np

class IOProvider:
    """
    Combines Input and Output from Input signal Strategy
    and Output Signal Strategy for use in ML Algorithms
    """

    ohlc_data: pd.DataFrame
    input_matrix: pd.DataFrame
    output_vector: np.array


    def __init__(self, symbol: str):
        self.ohlc_data = obtain_ohlc_data(symbol, include_all_indicators=True)
        # self.ohlc_data = obtain_ohlc_data(symbol, include_all_indicators=False)


    def obtain_input_matrix(self) -> np.array:
        ism = InputSignalMatrix(self.ohlc_data)
        self.input_matrix = ism.get_input_matrix()

        self.input_matrix.drop('Volume', axis=1, inplace=True)
        self.input_matrix.drop('MADCO-1', axis=1, inplace=True)
        self.input_matrix.drop('MADCO-3', axis=1, inplace=True)
        self.input_matrix.drop('BolBSB-2', axis=1, inplace=True)

        # return self.input_matrix
        return self.input_matrix.to_numpy()

    def obtain_output_vector(self, training_signal="get_pred_profit_signals"):
        osv = OutputSignalVector(self.ohlc_data, -1)
        self.output_vector = osv.signal_mux(training_signal)
        return self.output_vector[0], osv.start_end_set
