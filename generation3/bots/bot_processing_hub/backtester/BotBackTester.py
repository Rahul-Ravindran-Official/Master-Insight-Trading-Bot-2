from typing import List, Tuple

import numpy as np
import pandas as pd

from generation3.bots.bot_processing_hub.refined_data_for_bot import RefinedData
from generation3.ensemble_trading_bot import SignalProvider


class BotBackTester:

    # Inputs
    _price_adj_close: np.array
    _buy_signals: np.array
    _sell_signals: np.array
    _buy_treshold: Tuple[int, int]
    _sell_treshold: Tuple[int, int]
    _spread: float
    _lot_size: float
    _pip_definition: float
    _profit_per_pip_per_lot: float

    # Metrics
    _price_gained: float
    _profits_gained: float
    _days_traded: int
    _trades: int

    _profit_count: int
    _loss_count: int
    _max_profit: float
    _max_loss: float

    _loss_profit_tx: List[float]
    _binary_loss_profit_tx: List[float]
    _max_consecutive_profit: int
    _max_consecutive_loss: int

    def __init__(
            self,
            bot: SignalProvider,
            price_adj_close,
            buy_treshold=(0.5, 1.0),
            sell_treshold=(0.5, 1.0),
            spread=3.5,
            lot_size=0.1,
            pip_definition=0.0001,
            profit_per_pip_per_lot=10
    ):
        self._price_adj_close = price_adj_close
        self._buy_signals = bot.get_buy_signal()
        self._sell_signals = bot.get_sell_signal()

        self._price_gained = 0
        self._days_traded = 0
        self._trades = 0
        self._loss_count = 0
        self._profit_count = 0

        self._buy_treshold = buy_treshold
        self._sell_treshold = sell_treshold
        self._spread = spread
        self._lot_size = lot_size
        self._pip_definition = pip_definition
        self._profit_per_pip_per_lot = profit_per_pip_per_lot

        self._profits_gained = 0

        self._max_profit = -float("inf")
        self._max_loss = float("inf")

        self._loss_profit_tx = []
        self._max_consecutive_profit = 0
        self._max_consecutive_loss = 0

    def back_test(self):

        should_capture_metrics = True

        is_order_open = False
        buy_price = None
        days_open = 0

        for idx in range(len(self._price_adj_close)):

            days_open += 1

            price = self._price_adj_close[idx]
            buy_signal = self._buy_signals[idx]
            sell_signal = self._sell_signals[idx]

            if is_order_open and (self._sell_treshold[0] <= sell_signal <= self._sell_treshold[1]):

                if should_capture_metrics:
                    # Save Metrics
                    profit_made = price - buy_price - (self._spread * self._pip_definition)
                    self._price_gained += profit_made
                    self._days_traded += days_open
                    self._trades += 1

                    _max_profit: float
                    _max_loss: float
                    _max_consecutive_profit: int
                    _max_consecutive_loss: int

                    if profit_made > 0:
                        self._profit_count += 1
                        if profit_made > self._max_profit:
                            self._max_profit = profit_made
                    else:
                        self._loss_count += 1
                        if profit_made < self._max_loss:
                            self._max_loss = profit_made

                    self._loss_profit_tx.append(profit_made)
                else:
                    should_capture_metrics = True

                # Reset
                is_order_open = False
                buy_price = None
                days_open = 0




            if not is_order_open and (self._buy_treshold[0] <= buy_signal <= self._buy_treshold[1]):
                # Reset
                is_order_open = True
                buy_price = price
                days_open = 1


        # Loop Exit

        pips_gained = self._price_gained/self._pip_definition
        self._profits_gained = pips_gained * self._lot_size * self._profit_per_pip_per_lot

        numpy_loss_profit_tx = np.array(self._loss_profit_tx)
        self._binary_loss_profit_tx = numpy_loss_profit_tx[numpy_loss_profit_tx > 0]

        # tmp = pd.DataFrame(self._binary_loss_profit_tx)
        # tmp = tmp.ne(0)
        # self._max_consecutive_loss = tmp.cumsum()[~tmp].value_counts().max()
        #
        # tmp = pd.DataFrame(self._binary_loss_profit_tx)
        # tmp = tmp.ne(1)
        # self._max_consecutive_profit = tmp.cumsum()[~tmp].value_counts().max()

    def print_stats(self):
        print('Price Gained : {}'.format(self._price_gained))
        print('Pips Gained : {}'.format(self._price_gained/self._pip_definition))
        print('Profits Earned : {}'.format(self._profits_gained))
        print('Days Traded : {}'.format(self._days_traded))
        print('Total Trades : {}'.format(self._trades))
        print('Profit Counts : {}'.format(self._profit_count))
        print('Loss Count : {}'.format(self._loss_count))
        print('Max Price Profit : {}'.format(self._max_profit))
        print('Max Price Loss : {}'.format(self._max_loss))
        # print('Max Consecutive Profit : {}'.format(self._max_consecutive_profit))
        # print('Max Consecutive Loss : {}'.format(self._max_consecutive_loss))

    def get_stats(self):
        return {
            'price_gained': self._price_gained,
            'pips_gained': self._price_gained / self._pip_definition,
            'profits_earned': self._profits_gained,
            'days_traded': self._days_traded,
            'total_trades': self._trades,
            'profit_counts': self._profit_count,
            'loss_count': self._loss_count,
            'max_price_profit': self._max_profit,
            'max_price_loss': self._max_loss
        }
