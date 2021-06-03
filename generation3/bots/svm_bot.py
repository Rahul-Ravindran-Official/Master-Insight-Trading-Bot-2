import unittest
import numpy as np

from generation3.ensemble_trading_bot import SignalProvider


class SVMBot(SignalProvider):
    def get_signal(self) -> np.array:
        return SVMBot_V1().get_signal()


class SVMBot_V1(SignalProvider):
    def get_signal(self) -> np.array:
        pass


class SVMBotPlayground(unittest.TestCase):
    def test_sanity(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
