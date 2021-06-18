from typing import Dict, List

from generation4.bot_dataset import BotDataset
from generation4.bots.bot_template import TradingBot
import pandas as pd


class BotMetrics:

    tp_wins: int
    sl_loss: int
    s_trade_days_treshold: int
    win_trades: int
    loss_trades: int
    total_days_in_trade: int
    trade_tx: List[float]
    win_rate: float
    total_trades: int
    avg_trades_per_month: float
    total_tradable_days: int
    profit_pips_earned: float

    def __init__(self):
        self.tp_wins = 0
        self.sl_loss = 0
        self.s_trade_days_treshold = 0
        self.win_trades = 0
        self.loss_trades = 0
        self.total_days_in_trade = 0
        self.trade_tx = []
        self.win_rate = 0
        self.total_trades = 0
        self.avg_trades_per_month = 0
        self.total_tradable_days = 0
        self.profit_pips_earned = 0

    def metrics(self):
        return {
            'tp_wins': self.tp_wins,
            'sl_loss': self.sl_loss,
            's_trade_days_treshold': self.s_trade_days_treshold,
            'win_trades': self.win_trades,
            'loss_trades': self.loss_trades,
            'total_days_in_trade': self.total_days_in_trade,
            'trade_tx': self.trade_tx,
            'win_rate': self.win_rate,
            'total_trades': self.total_trades,
            'avg_trades_per_month': self.avg_trades_per_month,
            'total_tradable_days': self.total_tradable_days,
            'profit_pips_earned': self.profit_pips_earned
        }


class BotBackTester:

    # Input Parameters
    _risk_reward_ratio: float
    _minimum_avg_trades_per_month: float
    _minimum_win_rate: float
    _take_profit_pips: int
    _pip_definition: float
    _maximum_days_to_hold_per_trade: int
    _bot_confidence_treshold: float
    _spread: float

    # Bot
    _bot: TradingBot
    _bot_type: type(TradingBot)
    _bot_metadata: Dict
    bot_metrics: BotMetrics

    def __init__(
            self,
            minimum_avg_trades_per_month: float,
            minimum_win_rate: float,
            stop_loss_pips: int,
            take_profit_pips: int,
            pip_definition: float,
            maximum_days_to_hold_per_trade: int,
            bot_confidence_treshold: float,
            spread: float,
            bot_type: type(TradingBot),
            bot_metadata: Dict
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

        # Bot Data
        self._b_tr_x = pd.DataFrame()
        self._b_tr_y = pd.DataFrame()
        self._b_va_x = pd.DataFrame()
        self._b_va_y_full = pd.DataFrame()

        # Bot
        self._bot_type = bot_type
        self._bot_metadata = bot_metadata

        # Bot Metrics
        self.bot_metrics = BotMetrics()

    def execute(self):
        self.set_bot_dataset()
        self._bot = self._bot_type(
            self._b_tr_x,
            self._b_tr_y,
            self._b_va_x,
            self._bot_metadata,
        )
        self.bot_performance(self._bot, self._bot_confidence_treshold)
        return self.get_bot_metrics()

    def set_bot_dataset(self):
        dataset = BotDataset(
            pip_definition=self._pip_definition,
            take_profit_pips=self._take_profit_pips,
            stop_loss_pips=self._stop_loss_pips,
            spread=self._spread,
            maximum_days_to_hold_per_trade=self._maximum_days_to_hold_per_trade
        ).return_data()

        self._b_tr_x = dataset['b_tr_x']
        self._b_tr_y = dataset['b_tr_y']
        self._b_va_x = dataset['b_va_x']
        self._b_va_y_full = dataset['b_va_y_full']

    def bot_performance(self, bot: TradingBot, treshold_action: float):

        prediction = bot.predict()

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
                    self.bot_metrics.tp_wins += 1
                elif record['signal'] < 0:
                    self.bot_metrics.sl_loss += 1
                else:
                    self.bot_metrics.s_trade_days_treshold += 1

                if tx > 0:
                    win_trades += 1
                else:
                    loss_trades += 1

                trade_tx.append(tx)

                curr_idx += int(days_held)

        # Copy To Global Variables
        self.bot_metrics.trade_tx = trade_tx
        self.bot_metrics.total_trades = len(self.bot_metrics.trade_tx)

        self.bot_metrics.win_trades = win_trades
        self.bot_metrics.loss_trades = loss_trades

        self.bot_metrics.total_days_in_trade = total_days_in_trade

        if self.bot_metrics.total_trades == 0:
            self.bot_metrics.win_rate = 0
        else:
            self.bot_metrics.win_rate = self.bot_metrics.win_trades / self.bot_metrics.total_trades

        self.bot_metrics.total_tradable_days = len(price)
        self.bot_metrics.avg_trades_per_month = self.bot_metrics.total_trades / (self.bot_metrics.total_tradable_days/21)
        self.bot_metrics.profit_pips_earned = sum(self.bot_metrics.trade_tx) * (1 / self._pip_definition)

    def bot_stats(self):

        print(f'Risk Reward Ratio : {self._risk_reward_ratio}')

        print('')
        print(f'Win Trades : {self.bot_metrics.win_trades}')
        print(f'Loss Trade : {self.bot_metrics.loss_trades}')

        print('')
        print(f'TP Wins : {self.bot_metrics.tp_wins}')
        print(f'SL Losses : {self.bot_metrics.sl_loss}')

        print('')
        print(f'Total Days in Trades : {self.bot_metrics.total_days_in_trade}')
        print(f'Total Trades : {self.bot_metrics.total_trades}')
        print(f'Win Rate : {self.bot_metrics.win_rate}')
        print(f'Total Trading Days : {self.bot_metrics.total_tradable_days}')
        print(f'Trades every month (20 trading days) : {self.bot_metrics.win_rate}')
        print(f'Profit Pips Earned : {self.bot_metrics.profit_pips_earned}')


        trade_rate_satisfied = self.bot_metrics.avg_trades_per_month >= self._minimum_avg_trades_per_month
        win_rate_satisfied = self.bot_metrics.win_rate >= self._minimum_win_rate

        print('')
        print('Verdict')
        if trade_rate_satisfied and win_rate_satisfied:
            print('This Bot is Deployable')
        else:
            print(f'Bot is Undeployable : Trading Rate Satisfied → {trade_rate_satisfied} | Win Rate Satisfied → {win_rate_satisfied}')

    def get_bot_metrics(self) -> BotMetrics:

        return self.bot_metrics
