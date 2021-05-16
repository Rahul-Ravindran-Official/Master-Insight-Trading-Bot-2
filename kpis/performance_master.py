import numpy as np


class PerformanceMaster:
    """
        This class gives the performance of an algorithm
        with the inputs:
        1. Actual Market Data Vector: [a, b, c, d]
        2. Buy, Sell infused Vector: [1, 0, 0, 0, -1, 0, 1]
    """

    def __init__(self, price_closed: np.array, buy_sell_signals: np.array):
        self.price_closed = price_closed
        self.buy_sell_signals = buy_sell_signals

    def compute_yield(self):

        entry_price = -1
        ongoing_signal = None

        tx_yield = []

        for idx, signal in enumerate(self.buy_sell_signals):

            if signal == 0:
                continue

            # Base Case
            if ongoing_signal is None:
                ongoing_signal = signal
                entry_price = self.price_closed[idx]
                continue

            if ongoing_signal != signal:
                curr_price = self.price_closed[idx]

                if ongoing_signal == 1:
                    tx_yield.append(curr_price - entry_price)
                else:
                    tx_yield.append(entry_price - curr_price)

                # Change buy -> sell and vice-versa
                ongoing_signal = signal
                entry_price = curr_price

        return sum(tx_yield)


if __name__ == "__main__":
    print("Yield: " + str(PerformanceMaster([5, 6, 7, 8, 10, 9, 7, 5], [1, 0, 0, 0, -1, 0, 0, 1]).compute_yield()))
