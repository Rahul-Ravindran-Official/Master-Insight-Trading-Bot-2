import unittest
from typing import Dict, List, Tuple

from sklearn.neighbors import KNeighborsClassifier

from generation3.bots.bot_processing_hub.refined_data_for_bot import RefinedData
import pandas as pd
import numpy as np
from hyperopt import hp
from hyperopt import fmin, tpe


class CalculatedRiskBot:

    # Input Parameters
    _risk_reward_ratio: float
    _minimum_avg_trades_per_month: float
    _minimum_win_rate: float
    _take_profit_pips: int
    _pip_definition: float
    _maximum_days_to_hold_per_trade: int
    _bot_confidence_treshold: float
    _spread: float

    # Features
    _b_tr_x: pd.DataFrame
    _b_va_x: pd.DataFrame
    _b_te_x: pd.DataFrame

    # Targets
    _b_tr_y: pd.DataFrame
    _b_va_y: pd.DataFrame
    _b_te_y: pd.DataFrame

    _b_tr_y_full: pd.DataFrame
    _b_va_y_full: pd.DataFrame
    _b_te_y_full: pd.DataFrame

    # Bot Metrics
    _tp_wins: int
    _sl_loss: int
    _s_trade_days_treshold: int

    _win_trades: int
    _loss_trades: int
    _total_days_in_trade: int
    _trade_tx: List[float]
    _win_rate: float
    _total_trades: int
    _avg_trades_per_month: float
    _total_tradable_days: int
    _profit_pips_earned: float

    def __init__(
            self,
            minimum_avg_trades_per_month: float,
            minimum_win_rate: float,
            stop_loss_pips: int,
            take_profit_pips: int,
            pip_definition: float,
            maximum_days_to_hold_per_trade: int,
            bot_confidence_treshold: float,
            spread: float
    ):
        # Init Input Parameters
        self._minimum_avg_trades_per_month = minimum_avg_trades_per_month
        self._minimum_win_rate = minimum_win_rate
        self._stop_loss_pips = stop_loss_pips
        self._take_profit_pips = take_profit_pips
        self._pip_definition = pip_definition
        self._maximum_days_to_hold_per_trade = maximum_days_to_hold_per_trade
        self._bot_confidence_treshold = bot_confidence_treshold
        self._spread = spread
        self._risk_reward_ratio = self._take_profit_pips / self._stop_loss_pips

        # Init Predictors
        self._b_tr_x = pd.DataFrame()
        self._b_va_x = pd.DataFrame()
        self._b_te_x = pd.DataFrame()

        # Init Targets
        self._b_tr_y = pd.DataFrame()
        self._b_va_y = pd.DataFrame()
        self._b_te_y = pd.DataFrame()

        # Init Targets -> Days of Holding for triggering SL/TP
        self._b_tr_y_full = pd.DataFrame()
        self._b_va_y_full = pd.DataFrame()
        self._b_te_y_full = pd.DataFrame()

        # Bot Metrics
        self._win_trades = 0
        self._loss_trades = 0
        self._total_days_in_trade = 0
        self._trade_tx = []
        self._win_rate = 0
        self._total_trades = 0
        self._avg_trades_per_month = 0
        self._total_tradable_days = 0
        self._tp_wins = 0
        self._sl_loss = 0
        self._s_trade_days_treshold = 0
        self._profit_pips_earned = 0

    def chain_of_command(self):

        self.get_features()
        self.get_ideal_buy_signals()

        # Bots
        self.bot_performance_v2(self.knn_bot, self._bot_confidence_treshold)
        # self.bot_stats()

    def get_features(self):
        self._b_tr_x, _, self._b_va_x, _, self._b_te_x, _ = RefinedData.get_dataset_common(
            None,
            buy_sell_dataset='b'
        )

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
                    # buy_signal.append(0)
                    buy_signal.pop()
                    buy_signal.append([0, days_holding, profit_earned])
                    break

                if current_price >= take_profit_trigger_level:
                    # buy_signal.append(1)
                    buy_signal.pop()
                    buy_signal.append([1, days_holding, profit_earned])
                    break
                elif current_price <= take_loss_trigger_level:
                    # buy_signal.append(-1)
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

    def knn_bot(self):
        neigh = KNeighborsClassifier(
            n_neighbors=5
        )
        neigh.fit(self._b_tr_x, self._b_tr_y)
        return neigh.predict_proba(self._b_va_x.to_numpy())[:, 1]

    def bot_performance(self, bot: callable, treshold_action: float, spread: float):

        prediction = bot()

        # Metrics
        win_trades = 0
        loss_trades = 0
        total_days_in_trade = 0
        trade_tx = []

        price = self._b_va_x['Adj Close'].to_numpy()

        curr_idx = 0

        while curr_idx < len(price):

            if prediction[curr_idx] >= treshold_action:

                days_held = self._b_va_y_full[curr_idx]
                total_days_in_trade += days_held

                price_today = price[curr_idx]
                price_sell = price[curr_idx + days_held] # We already know how long to hold for to hit take profit!

                # Take into account spread!
                tx = price_sell - price_today - spread

                if tx > 0:
                    win_trades += 1
                else:
                    loss_trades += 1

                trade_tx.append(tx)

                curr_idx += days_held

            curr_idx += 1

        # Copy To Global Variables
        self._trade_tx = trade_tx
        self._total_trades = len(self._trade_tx)
        self._win_trades = win_trades
        self._loss_trades = loss_trades
        self._total_days_in_trade = total_days_in_trade
        self._win_rate = self._win_trades/self._total_trades
        self._total_tradable_days = len(price)
        self._avg_trades_per_month = self._total_trades / (self._total_tradable_days/21)

    def bot_performance_v2(self, bot: callable, treshold_action: float):
        prediction = bot()

        # Metrics
        win_trades = 0
        loss_trades = 0
        total_days_in_trade = 0
        trade_tx = []

        price = self._b_va_x['Adj Close'].to_numpy()

        curr_idx = -1
        while curr_idx+1 < len(price):
            curr_idx += 1
            if prediction[curr_idx] >= treshold_action:
                record = self._b_va_y_full.iloc[curr_idx]

                days_held = record['days_holding']
                total_days_in_trade += days_held

                # Take into account spread!
                tx = record['profit_earned']

                if record['signal'] > 0:
                    self._tp_wins += 1
                elif record['signal'] < 0:
                    self._sl_loss += 1
                else:
                    self._s_trade_days_treshold += 1

                if tx > 0:
                    win_trades += 1
                else:
                    loss_trades += 1

                trade_tx.append(tx)

                curr_idx += int(days_held)

        # Copy To Global Variables
        self._trade_tx = trade_tx
        self._total_trades = len(self._trade_tx)

        self._win_trades = win_trades
        self._loss_trades = loss_trades

        self._total_days_in_trade = total_days_in_trade

        if self._total_trades == 0:
            self._win_rate = 0
        else:
            self._win_rate = self._win_trades/self._total_trades

        self._total_tradable_days = len(price)
        self._avg_trades_per_month = self._total_trades / (self._total_tradable_days/21)
        self._profit_pips_earned = sum(self._trade_tx) * (1 / self._pip_definition)

    def objective_function(self):
        """
        :return: Lower is better trade
        """
        # -((self._tp_wins * self._take_profit_pips) - (self._sl_loss * self._stop_loss_pips))
        return -self._profit_pips_earned

    def bot_stats(self):

        print(f'Risk Reward Ratio : {self._risk_reward_ratio}')

        print('')
        print(f'Win Trades : {self._win_trades}')
        print(f'Loss Trade : {self._loss_trades}')

        print('')
        print(f'TP Wins : {self._tp_wins}')
        print(f'SL Losses : {self._sl_loss}')

        print('')
        print(f'Total Days in Trades : {self._total_days_in_trade}')
        print(f'Total Trades : {self._total_trades}')
        print(f'Win Rate : {self._win_rate}')
        print(f'Total Trading Days : {self._total_tradable_days}')
        print(f'Trades every month (20 trading days) : {self._win_rate}')
        print(f'Profit Pips Earned : {self._profit_pips_earned}')


        trade_rate_satisfied = self._avg_trades_per_month >= self._minimum_avg_trades_per_month
        win_rate_satisfied = self._win_rate >= self._minimum_win_rate

        print('')
        print('Verdict')
        if trade_rate_satisfied and win_rate_satisfied:
            print('This Bot is Deployable')
        else:
            print(f'Bot is Undeployable : Trading Rate Satisfied → {trade_rate_satisfied} | Win Rate Satisfied → {win_rate_satisfied}')

    def bot_visualization(self):
        pass


#unittest.TestCase
class CalculatedRiskBotPlayground:

    def test_something(self):
        self.assertEqual(True, False)

    @staticmethod
    def test_crb(args: Dict):

        # args = {
        #     'stop_loss_pips': 10,
        #     'take_profit_pips': 100,
        #     'maximum_days_to_hold_per_trade': 5,
        #     'bot_confidence_treshold': 0.5
        # }

        crb = CalculatedRiskBot(
            minimum_avg_trades_per_month=5,
            minimum_win_rate=0.1,
            stop_loss_pips=args['stop_loss_pips'],
            take_profit_pips=args['take_profit_pips'],
            pip_definition=0.0001,
            maximum_days_to_hold_per_trade=args['maximum_days_to_hold_per_trade'],
            bot_confidence_treshold=args['bot_confidence_treshold'],
            spread=0
        )

        crb.chain_of_command()

        if 'stats' in args:
            crb.bot_stats()

        return crb.objective_function()

class BotOptimizer:

    @staticmethod
    def objective(args: Dict) -> float:
        bot = CalculatedRiskBot(
            minimum_avg_trades_per_month=5,
            minimum_win_rate=0.1,
            stop_loss_pips=args['stop_loss_pips'],
            take_profit_pips=args['take_profit_pips'],
            pip_definition=0.0001,
            maximum_days_to_hold_per_trade=args['maximum_days_to_hold_per_trade'],
            bot_confidence_treshold=0.5,
            spread=0
        )
        bot.chain_of_command()
        v = bot.objective_function()
        print(f'Obj -> {v}')
        return v

    @staticmethod
    def optimize():

        space = {
            'stop_loss_pips': hp.randint('stop_loss_pips', 200) + 1,
            'take_profit_pips': hp.randint('take_profit_pips', 200) + 1,
            'maximum_days_to_hold_per_trade': hp.randint('maximum_days_to_hold_per_trade', 10) + 1,
            'bot_confidence_treshold': hp.uniform('bot_confidence_treshold', 0, 1)
        }

        best = fmin(
            BotOptimizer.objective,
            space,
            algo=tpe.suggest,
            max_evals=100,

        )

        print(best)
        print(BotOptimizer.objective(best))


if __name__ == '__main__':

    BotOptimizer.optimize()

    # args = {
    #     'bot_confidence_treshold': 0.6972694716982044,
    #     'maximum_days_to_hold_per_trade': 1,
    #     'stop_loss_pips': 10,
    #     'take_profit_pips': 100
    # }
    #
    # args['stats'] = 1
    #
    # CalculatedRiskBotPlayground.test_crb(args)
