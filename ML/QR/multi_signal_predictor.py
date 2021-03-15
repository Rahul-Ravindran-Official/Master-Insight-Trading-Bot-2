import math
from typing import Tuple

import numpy as np
from sklearn.preprocessing import minmax_scale


class MultiSignalPredictor:
    """
    The purpose of this class is to reduce the dimension of an
    m*n to a n*n square matrix where m > n

    Prediction Signals For Input: [Count => 25]
    + MA                : 10
    + MA Cross-Overs    : 5
    + MACD              : 2
    + ATR               : 1
    + Bollinger Bands   : 1
    + OBV               : 2
    + RSI               : 2
    + Slope             : 1
    + ADX               : 1

    """

    # Settings
    train_to_test_ratio: float

    input_signals: np.matrix
    output_signal: np.matrix

    input_signals_train: np.matrix
    output_signal_train: np.matrix

    input_signals_test: np.matrix
    output_signal_test: np.matrix

    qr_factorised_input_signals: np.matrix
    qr_factorised_output_signals: np.matrix

    generate_prediction_matrix: np.matrix

    prediction_cutoff: float

    def __init__(
            self, input_signal: np.matrix,
            output_signal: np.matrix, train_to_test_ratio: float,
            prediction_cutoff: float
    ):
        self.input_signals = input_signal
        self.output_signal = output_signal
        self.train_to_test_ratio = train_to_test_ratio
        self.prediction_cutoff = prediction_cutoff

    def compute_magic_numbers(self):
        self.split_train_test()
        self.perform_qr_factorization()
        self.performance_residual = HouseholderUtility.find_residual(
            self.qr_factorised_input_signals,
            self.qr_factorised_output_signals,
            self.qr_magic_numbers
        )
        print("Performance Residual : " + str(self.performance_residual))

        self.generate_prediction_vector()

        self.generate_prediction_signal()

    def split_train_test(self):
        total_rows = self.input_signals.shape[0]
        train_rows = int(math.floor(self.train_to_test_ratio * total_rows))

        # Input Rows Split
        self.input_signals_train = self.input_signals[0:train_rows]
        self.input_signals_test = self.input_signals[train_rows:]

        # Output Rows Split
        self.output_signals_train = self.output_signal[0:train_rows]
        self.output_signals_test = self.output_signal[train_rows:]

    def perform_qr_factorization(self):
        self.qr_magic_numbers, \
        self.qr_factorised_input_signals, \
        self.qr_factorised_output_signals = HouseholderUtility.solve_qr_householder(
                self.input_signals_train, self.output_signals_train
        )

    def generate_prediction_vector(self):
        self.prediction_vector = np.matmul(self.input_signals_train, self.qr_magic_numbers)
        self.prediction_vector_test = np.matmul(self.input_signals_test, self.qr_magic_numbers)

    def predict_signal_row(self, input_signal: np.array) -> float:
        confidence = float(np.matmul(input_signal, self.qr_magic_numbers))
        return confidence >= self.prediction_cutoff

    def generate_prediction_signal(self):

        pv_train_normalised = minmax_scale(
            self.prediction_vector,
            feature_range=(0, 1),
            axis=0,
            copy=True
        )

        pv_test_normalised = minmax_scale(
            self.prediction_vector_test,
            feature_range=(0, 1),
            axis=0,
            copy=True
        )

        self.prediction_vector_signal = (pv_train_normalised >= self.prediction_cutoff).astype(np.float32)
        self.prediction_vector_test_signal = (pv_test_normalised >= self.prediction_cutoff).astype(np.float32)

class HouseholderUtility:

    @staticmethod
    def householder_v(a):
        """Return the vector $v$ that defines the Householder Transform
             H = I - 2 np.matmul(v, v.T) / np.matmul(v.T, v)
        that will eliminate all but the first element of the
        input vector a. Choose the $v$ that does not result in
        cancellation.

        Do not modify the vector `a`.

        Example:
            >>> a = np.array([2., 1., 2.])
            >>> HouseholderUtility.householder_v(a)
            array([5., 1., 2.])
            >>> a
            array([2., 1., 2.])
        """
        b = a.copy()
        alpha = 0
        b_0_sign = 1

        for c in range(a.shape[0]):
            alpha += math.pow(a[c], 2)
        alpha = math.sqrt(alpha)

        if b[0] < 0:
            b_0_sign = -1

        alpha = -b_0_sign * alpha

        b[0] = b[0] - alpha

        return b

    @staticmethod
    def apply_householder(v, u):
        """Return the result of the Householder transformation defined
        by the vector $v$ applied to the vector $u$. You should not
        compute the Householder matrix H directly.

        Example:

        >>> HouseholderUtility.apply_householder(np.array([5., 1., 2.]), np.array([2., 1., 2.]))
        array([-3.,  0.,  0.])
        >>> HouseholderUtility.apply_householder(np.array([5., 1., 2.]), np.array([2., 3., 4.]))
        array([-5. ,  1.6,  1.2])
        """
        return u - (2 * ((np.matmul(v.transpose(), u)) / (
            np.matmul(v.transpose(), v))) * v)

    @staticmethod
    def apply_householder_matrix(v, U):
        """Return the result of the Householder transformation defined
        by the vector $v$ applied to all the vectors in the matrix U.
        You should not compute the Householder matrix H directly.

        Example:

        >>> v = np.array([5., 1., 2.])
        >>> U = np.array([[2., 2.],[1., 3.],[2., 4.]])
        >>> HouseholderUtility.apply_householder_matrix(v, U)
        array([[-3. , -5. ],
               [ 0. ,  1.6],
               [ 0. ,  1.2]])
        """
        i = 2 * (np.outer((np.matmul(v.T, U) / np.matmul(v.T, v)), v))
        return U - i.T

    @staticmethod
    def qr_householder(A, b):
        """Return the matrix [R O]^T, and vector [c1 c2]^T equivalent
        to the system $Ax \approx b$. This algorithm is similar to
        Algorithm 3.1 in the textbook.
        """
        for k in range(A.shape[1]):
            v = HouseholderUtility.householder_v(A[k:, k])
            if np.linalg.norm(v) != 0:
                A[k:, k:] = HouseholderUtility.apply_householder_matrix(v, A[k:,
                                                                           k:])
                b[k:] = HouseholderUtility.apply_householder(v, b[k:])
        # now, A is upper-triangular
        return A, b

    @staticmethod
    def back_substitution(A, b):
        """Return a vector x with np.matmul(A, x) == b, where
            * A is an nxn numpy matrix that is upper-triangular and non-singular
            * b is an nx1 numpy vector
        >>> A = np.array([[2., 1.],[0., -2.]])
        >>> b = np.array([1., 2.])
        >>> HouseholderUtility.back_substitution(A, b)
        array([ 1 , -1])

        """
        n = A.shape[0]
        x = np.zeros_like(b, dtype=float)
        for i in range(n - 1, -1, -1):
            s = 0
            for j in range(n - 1, i, -1):
                s += A[i, j] * x[j]
            x[i] = (b[i] - s) / A[i, i]
        return x

    @staticmethod
    def solve_qr_householder(A, b) -> Tuple[np.matrix, np.matrix, np.matrix]:
        """
        Return the solution x to the linear least squares problem
            $$Ax \approx b$$ using Householder QR decomposition.
        Where A is an (m x n) matrix, with m > n, rank(A) = n, and
              b is a vector of size (m)
        """

        cpy_A = A.copy()
        cpy_b = b.copy()

        HouseholderUtility.qr_householder(cpy_A, cpy_b)
        new_A = cpy_A[0:cpy_A.shape[1], 0:cpy_A.shape[1]]
        new_b = cpy_b[0:cpy_A.shape[1]]
        r = HouseholderUtility.back_substitution(new_A, new_b)
        return r.T, new_A, new_b

    @staticmethod
    def find_residual(A: np.matrix, b: np.matrix, r: np.matrix) -> float:
        return math.sqrt(sum(np.square(b - np.matmul(A, r))))

