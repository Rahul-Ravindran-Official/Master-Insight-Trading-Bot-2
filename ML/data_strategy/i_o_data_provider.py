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


    def obtain_input_matrix(self) -> np.array:
        ism = InputSignalMatrix(self.ohlc_data)
        self.input_matrix = ism.get_input_matrix()
        return self.input_matrix.to_numpy()

    def obtain_output_vector(self):
        osv = OutputSignalVector(self.ohlc_data, -1)
        self.output_vector = osv.get_buy_signals()
        return self.output_vector[0]
