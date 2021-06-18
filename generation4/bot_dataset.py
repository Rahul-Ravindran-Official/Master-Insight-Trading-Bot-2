from generation3.bots.bot_processing_hub.refined_data_for_bot import RefinedData

import pandas as pd
import numpy as np


class BotDataset:

    # Input Parameters
    _pip_definition: float
    _take_profit_pips: int
    _stop_loss_pips: int
    _spread: float
    _maximum_days_to_hold_per_trade: int

    # Buy Dataset Predictors
    _b_tr_x: pd.DataFrame
    _b_va_x: pd.DataFrame
    _b_te_x: pd.DataFrame
    # Buy Dataset Targets
    _b_tr_y: pd.DataFrame
    _b_va_y: pd.DataFrame
    _b_te_y: pd.DataFrame
    _b_tr_y_full: pd.DataFrame
    _b_va_y_full: pd.DataFrame
    _b_te_y_full: pd.DataFrame
    # Sell Dataset Predictors
    _s_tr_x: pd.DataFrame
    _s_va_x: pd.DataFrame
    _s_te_x: pd.DataFrame
    _s_tr_y: pd.DataFrame
    _s_va_y: pd.DataFrame
    _s_te_y: pd.DataFrame
    _s_tr_y_full: pd.DataFrame
    _s_va_y_full: pd.DataFrame
    _s_te_y_full: pd.DataFrame


    def __init__(
            self,
            pip_definition: float,
            take_profit_pips: int,
            stop_loss_pips: int,
            spread: float,
            maximum_days_to_hold_per_trade: int,
    ):
        self._pip_definition = pip_definition
        self._take_profit_pips = take_profit_pips
        self._stop_loss_pips = stop_loss_pips
        self._spread = spread
        self._maximum_days_to_hold_per_trade = maximum_days_to_hold_per_trade

        # Dataset Buy
        self._b_tr_x = pd.DataFrame()
        self._b_va_x = pd.DataFrame()
        self._b_te_x = pd.DataFrame()

        self._b_tr_y = pd.DataFrame()
        self._b_va_y = pd.DataFrame()
        self._b_te_y = pd.DataFrame()
        self._b_tr_y_full = pd.DataFrame()
        self._b_va_y_full = pd.DataFrame()
        self._b_te_y_full = pd.DataFrame()

        # Dataset Sell
        self._s_tr_x = pd.DataFrame()
        self._s_va_x = pd.DataFrame()
        self._s_te_x = pd.DataFrame()

        self._s_tr_y = pd.DataFrame()
        self._s_va_y = pd.DataFrame()
        self._s_te_y = pd.DataFrame()
        self._s_tr_y_full = pd.DataFrame()
        self._s_va_y_full = pd.DataFrame()
        self._s_te_y_full = pd.DataFrame()

        self.get_features()
        self.get_ideal_buy_signals()
        self.get_ideal_sell_signals()


    def get_features(self):
        self._b_tr_x, _, self._b_va_x, _, self._b_te_x, _ = RefinedData.get_dataset_common(
            None,
            buy_sell_dataset='b'
        )

        self._s_tr_x, _, self._s_va_x, _, self._s_te_x, _ = RefinedData.get_dataset_common(
            None,
            buy_sell_dataset='s'
        )

    # ---------------
    # Buy Signals
    # ---------------

    def generate_ideal_signals_buy(self, predictor_block: pd.DataFrame) -> pd.DataFrame:

        take_profit_price = self._pip_definition * self._take_profit_pips
        stop_loss_price = self._pip_definition * self._stop_loss_pips

        price = predictor_block['Adj Close'].to_numpy()
        price_idx = list(range(len(price)))

        buy_signal = []

        for i in price_idx:

            buy_price = price[i]
            take_profit_trigger_level = buy_price + (take_profit_price + self._spread)
            take_loss_trigger_level = buy_price - (stop_loss_price - self._spread)
            days_holding = 0

            # Append By Default
            buy_signal.append([0, 1, 0])

            for j in price_idx[i+1:]:

                days_holding += 1

                current_price = price[j]

                profit_earned = current_price - buy_price

                if days_holding > self._maximum_days_to_hold_per_trade:
                    buy_signal.pop()
                    buy_signal.append([0, days_holding, profit_earned])
                    break

                if current_price >= take_profit_trigger_level:
                    buy_signal.pop()
                    buy_signal.append([1, days_holding, profit_earned])
                    break
                elif current_price <= take_loss_trigger_level:
                    buy_signal.pop()
                    buy_signal.append([-1, days_holding, profit_earned])
                    break

        np_buy_signal = np.array(buy_signal)

        return pd.DataFrame({
            'signal': np_buy_signal[:, 0],
            'days_holding': np_buy_signal[:, 1],
            'profit_earned': np_buy_signal[:, 2]
        })

    def get_ideal_buy_signals(self):
        b_tr_y = self.generate_ideal_signals_buy(self._b_tr_x)
        b_va_y = self.generate_ideal_signals_buy(self._b_va_x)
        b_te_y = self.generate_ideal_signals_buy(self._b_te_x)

        self._b_tr_y = b_tr_y['signal']
        self._b_va_y = b_va_y['signal']
        self._b_te_y = b_te_y['signal']

        self._b_tr_y_full = b_tr_y
        self._b_va_y_full = b_va_y
        self._b_te_y_full = b_te_y

    # ---------------
    # Sell Signals
    # ---------------

    def generate_ideal_signals_sell(self, predictor_block: pd.DataFrame) -> pd.DataFrame:

        take_profit_price = self._pip_definition * self._take_profit_pips
        stop_loss_price = self._pip_definition * self._stop_loss_pips

        price = predictor_block['Adj Close'].to_numpy()
        price_idx = list(range(len(price)))

        sell_signal = []

        for i in price_idx:

            buy_price = price[i]
            take_profit_trigger_level = buy_price - (take_profit_price + self._spread)
            take_loss_trigger_level = buy_price + (stop_loss_price - self._spread)
            days_holding = 0

            # Append By Default
            sell_signal.append([0, 1, 0])

            for j in price_idx[i+1:]:

                days_holding += 1

                current_price = price[j]

                profit_earned = buy_price - current_price

                if days_holding > self._maximum_days_to_hold_per_trade:
                    sell_signal.pop()
                    sell_signal.append([0, days_holding, profit_earned])
                    break

                if current_price <= take_profit_trigger_level:
                    sell_signal.pop()
                    sell_signal.append([1, days_holding, profit_earned])
                    break
                elif current_price >= take_loss_trigger_level:
                    sell_signal.pop()
                    sell_signal.append([-1, days_holding, profit_earned])
                    break

        np_sell_signal = np.array(sell_signal)

        return pd.DataFrame({
            'signal': np_sell_signal[:, 0],
            'days_holding': np_sell_signal[:, 1],
            'profit_earned': np_sell_signal[:, 2]
        })

    def get_ideal_sell_signals(self):
        s_tr_y = self.generate_ideal_signals_sell(self._s_tr_x)
        s_va_y = self.generate_ideal_signals_sell(self._s_va_x)
        s_te_y = self.generate_ideal_signals_sell(self._s_te_x)

        self._s_tr_y = s_tr_y['signal']
        self._s_va_y = s_va_y['signal']
        self._s_te_y = s_te_y['signal']

        self._s_tr_y_full = s_tr_y
        self._s_va_y_full = s_va_y
        self._s_te_y_full = s_te_y

    # ---------------
    # Output
    # ---------------

    def return_data(self):
        return {
            'b_tr_x': self._b_tr_x,
            'b_va_x': self._b_va_x,
            'b_te_x': self._b_te_x,
            'b_tr_y': self._b_tr_y,
            'b_va_y': self._b_va_y,
            'b_te_y': self._b_te_y,
            'b_tr_y_full': self._b_tr_y_full,
            'b_va_y_full': self._b_va_y_full,
            'b_te_y_full': self._b_te_y_full,
            's_tr_x': self._s_tr_x,
            's_va_x': self._s_va_x,
            's_te_x': self._s_te_x,
            's_tr_y': self._s_tr_y,
            's_va_y': self._s_va_y,
            's_te_y': self._s_te_y,
            's_tr_y_full': self._s_tr_y_full,
            's_va_y_full': self._s_va_y_full,
            's_te_y_full': self._s_te_y_full
        }
