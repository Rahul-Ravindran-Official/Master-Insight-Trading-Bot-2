from typing import Callable, Dict, List

import pandas as pd
import matplotlib.pyplot as plt

from backtesting.analytics.analytics_manager import AnalyticsManager
from backtesting.signal_processor.signal_manager import SignalManager
from market_data.ohlc_data import obtain_ohlc_data
from shared.BBHSSSignal import BBHSSSignal
from shared.BHSSignal import BHSSignal
from shared.Strategy import Strategy
import numpy as np


class BackTester:

    ohlc_data: pd.DataFrame
    current_trade: BHSSignal
    current_trade_executed: bool
    strategies: Dict[Strategy, float]
    analytics: AnalyticsManager
    risk: Dict[str, Callable[[float], float]]  # enter_buy, enter_sell, exit_buy, exit_sell

    def __init__(
            self,
            commodity: str,
            strategies: Dict[Strategy, float],
            risk: Dict[str, Callable[[float], float]]
    ):
        self.ohlc_data = obtain_ohlc_data(commodity)
        self.current_trade = BHSSignal.hold
        self.current_trade_executed = True
        self.strategies = strategies
        self.strategy_signal_columns = []
        self.analytics = AnalyticsManager(commodity)
        self.risk = risk

    def run(self):

        last_signal: BBHSSSignal = BBHSSSignal.no_action.value

        self.ohlc_data = SignalManager(self.ohlc_data, self.strategies).get_master_signal()

        self.init_columns()

        for i, signal in enumerate(self.ohlc_data["master_signal"]):

            if last_signal == BBHSSSignal.no_action.value:
                last_signal = signal
                continue

            if self.current_trade == BHSSignal.hold:
                # Not in trade
                self.not_in_trade(signal, last_signal)
            else:
                # In Trade
                self.in_trade(signal, last_signal, i)

            if not self.current_trade_executed:
                self.trade_executor(i)
            elif self.current_trade != BHSSignal.hold:
                self.returns_keeper(i)

            self.analytics.analyse_record(self.ohlc_data.loc[self.ohlc_data.index[i]])

            last_signal = signal

        self.ohlc_data[['cum_ret']] = self.ohlc_data[['cum_ret']].fillna(method='ffill')

        self.visualize_pct_return()

        print(self.analytics)
        return self.get_pct_return(), self.trades_made()

    def init_columns(self):
        self.ohlc_data['daily_ret'] = np.nan
        self.ohlc_data['cum_ret'] = np.nan
        self.ohlc_data['Trade'] = np.nan

    def in_trade(self, signal, last_signal, i):
        if self.current_trade == BHSSignal.buy and self.buy_exit_condition(signal, last_signal):
            # print("exited")
            self.current_trade = BHSSignal.hold
            self.not_in_trade(signal, last_signal)
        if self.current_trade == BHSSignal.sell and self.sell_exit_condition(signal, last_signal):
            # print("exited")
            self.current_trade = BHSSignal.hold
            self.not_in_trade(signal, last_signal)

    def not_in_trade(self, signal, last_signal):
        if self.sell_condition(signal, last_signal):
            # print("sell_condition triggered")
            self.trigger_sell_trade()
        if self.buy_condition(signal, last_signal):
            # print("buy_condition triggered")
            self.trigger_buy_trade()

    def trigger_buy_trade(self):
        self.current_trade = BHSSignal.buy
        self.current_trade_executed = False

    def trigger_sell_trade(self):
        self.current_trade = BHSSignal.sell
        self.current_trade_executed = False

    def trade_executor(self, i: int):
        if not self.current_trade_executed:
            self.ohlc_data.loc[self.ohlc_data.index[i], 'Trade'] = self.current_trade
            self.current_trade_executed = True

    def returns_keeper(self, i: int):
        if self.current_trade == BHSSignal.buy:

            self.ohlc_data.loc[
                self.ohlc_data.index[i],
                'daily_ret'
            ] = (self.ohlc_data['Close'][i] / self.ohlc_data['Close'][i-1]) - 1

            self.ohlc_data.loc[self.ohlc_data.index[i], 'cum_ret'] = self.ohlc_data[['daily_ret']].dropna().to_numpy().sum()

        elif self.current_trade == BHSSignal.sell:

            self.ohlc_data.loc[
                self.ohlc_data.index[i],
                'daily_ret'
            ] = (self.ohlc_data['Close'][i-1] / self.ohlc_data['Close'][i]) - 1

            self.ohlc_data.loc[self.ohlc_data.index[i], 'cum_ret'] = self.ohlc_data[['daily_ret']].dropna().to_numpy().sum()

    def sell_exit_condition(self, current_signal: float, last_signal: float):
        return self.risk['exit_sell'](current_signal)

    def sell_condition(self, current_signal: float, last_signal: float):
        return self.risk['enter_sell'](current_signal)

    def buy_condition(self, current_signal: float, last_signal: float):
        return self.risk['enter_buy'](current_signal)

    def buy_exit_condition(self, current_signal: float, last_signal: float):
        return self.risk['exit_buy'](current_signal)

    def get_pct_return(self):
        return self.ohlc_data[['cum_ret']].iloc[-1][0]

    def visualize_pct_return(self):
        self.ohlc_data[["cum_ret",'master_signal']].plot()
        plt.show()

    def trades_made(self):
        return len(self.ohlc_data.get('Trade').dropna())
