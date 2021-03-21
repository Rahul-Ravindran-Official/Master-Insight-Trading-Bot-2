from typing import Tuple

import numpy as np
from sklearn.preprocessing import minmax_scale

class DataNormalization:

    @staticmethod
    def normalize_column_0_1(data: np.array, col: int):
        return DataNormalization.normalize_utility(data, (0, 1), col)

    @staticmethod
    def normalize_column_minus_1_1(data: np.array, col: int):
        return DataNormalization.normalize_utility(data, (-1, 1), col)

    @staticmethod
    def normalize_all_columns_matrix_0_1(data: np.array):
        columns = data.shape[1]
        for i in range(columns):
            data[:, i] = DataNormalization.normalize_column_0_1(data, i)
        return data

    @staticmethod
    def normalize_utility(
            data: np.array,
            n_range: Tuple[int, int],
            col: int = None
    ):

        column_to_normalise = None

        if col is None:
            column_to_normalise = data
        else:
            column_to_normalise = data[:, col]

        return minmax_scale(
            column_to_normalise,
            feature_range=n_range,
            axis=0,
            copy=True
        )



