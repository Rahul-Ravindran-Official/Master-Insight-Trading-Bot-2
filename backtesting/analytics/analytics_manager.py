from typing import List
import pandas as pd
import numpy as np


class AnalyticsManager:

    total_trades: int
    total_success_trades: int
    total_fail_trades: int
    success_trades_pct: float
    fail_trades_pct: float

    pct_average_daily_return: float
    pct_average_monthly_return: float
    pct_average_yearly_return: float

    cumulative_return: float

    each_trade_proceeding: List[float]

    max_consecutive_win_trades: int
    max_consecutive_lose_trades: int

    largest_profit_trade_value: float
    largest_loss_trade_value: float

    avg_profit_trade_value: float
    avg_loss_trade_value: float

    commodity: str

    # Helper Variables
    _curr_streak_trades: int
    _curr_streak_dir: chr
    _last_ret: np.nan
    _current_trade_candle_ret: List[float]

    def __init__(self, commodity: str):
        self.total_trades = 0
        self.total_success_trades = 0
        self.total_fail_trades = 0
        self.success_trades_pct = 0
        self.fail_trades_pct = 0

        self.pct_average_daily_return = 0
        self.pct_average_monthly_return = 0
        self.pct_average_yearly_return = 0

        self.cumulative_return = 0

        self.each_trade_proceeding = []

        self.max_consecutive_win_trades = 0
        self.max_consecutive_lose_trades = 0

        self.largest_profit_trade_value = float('-inf')
        self.largest_loss_trade_value = float('inf')

        self.avg_profit_trade_value = 0
        self.avg_loss_trade_value = 0

        self.commodity = commodity

        # Helper Variables
        self._curr_streak_trades = 0
        self._curr_streak_dir = '~'
        self._last_ret = np.nan
        self._current_trade_candle_ret = []

    def analyse_record(self, record: pd.Series):
        """
        This method is to be called in a loop of the OHLC dataframe
        :param record: A numpy array of the current record to be analysed
        :return: None
        """
        curr_ret = record['daily_ret']

        if not np.isnan(curr_ret) and np.isnan(self._last_ret):

            self.total_trades += 1
            self._re_init_trade_variables()

        if np.isnan(curr_ret) and not np.isnan(self._last_ret):

            total_trade_ret = sum(self._current_trade_candle_ret)
            self.each_trade_proceeding.append(total_trade_ret)

            winning_trade = 1 if total_trade_ret>0 else 0

            self.total_success_trades += winning_trade
            self.total_fail_trades += 1 - winning_trade

            self.largest_profit_trade_value = \
                self._compute_largest_profit_trade(total_trade_ret)

            self.largest_loss_trade_value = \
                self._compute_largest_loss_trade(total_trade_ret)

            self.fail_trades_pct = self._compute_fail_trades_pct()
            self.success_trades_pct = self._compute_success_trades_pct()

            self._compute_winning_loosing_streak(winning_trade)

            self.avg_profit_trade_value = sum([e if e>0 else 0 for e in self.each_trade_proceeding])/(1 if self.total_success_trades == 0 else self.total_success_trades)
            self.avg_loss_trade_value = sum([e if e <= 0 else 0 for e in self.each_trade_proceeding])/(1 if self.total_fail_trades == 0 else self.total_fail_trades)

            self.cumulative_return = sum(self.each_trade_proceeding)

            self._re_init_trade_variables()

        self._current_trade_candle_ret.append(curr_ret)
        self._last_ret = curr_ret

    def _re_init_trade_variables(self):
        self._current_trade_candle_ret.clear()

    def _compute_winning_loosing_streak(self, winning_trade: int):
        """
        Manages the current longest running streak for wins and losses
        :param winning_trade: 1 if winning trade else 0
        :return: None - directly modifies instance variables
        """
        if winning_trade == 1:
            if self._curr_streak_dir != '+':
                self._curr_streak_dir = '+'
                self._curr_streak_trades = 1
            else:
                self._curr_streak_trades += 1

            self.max_consecutive_win_trades = max(
                self.max_consecutive_win_trades, self._curr_streak_trades)

        else:

            if self._curr_streak_dir != '-':
                self._curr_streak_dir = '-'
                self._curr_streak_trades = 1
            else:
                self._curr_streak_trades += 1

            self.max_consecutive_lose_trades = max(
                self.max_consecutive_lose_trades, self._curr_streak_trades)

    def _compute_success_trades_pct(self):
        return self.total_success_trades / self.total_trades

    def _compute_fail_trades_pct(self):
        return self.total_fail_trades / self.total_trades

    def _compute_largest_profit_trade(self, curr_val: float):
        """
        :param curr_val: the trade value
        :return: the largest profit trade value
        """
        return max(curr_val, self.largest_profit_trade_value)

    def _compute_largest_loss_trade(self, curr_val: float):
        """
        :param curr_val: the trade value
        :return: the largest loss trade value
        """
        return min(curr_val, self.largest_loss_trade_value)

    def __str__(self):
        return (
                "--------------------{commodity}--------------------" + "\n" +
                "Cumulative Return          : {cum_ret}" + "\n" +
                "Total Trades               : {tot_trades}" + "\n" +
                "Total Winning Trades       : {tot_succ_trades}" + "\n" +
                "Total Losing Trades        : {tot_fail_trad}" + "\n" +
                "Avg. Profit Trade          : {avg_profit_trade}" + "\n" +
                "Avg. Loss Trade            : {avg_loss_trade}" + "\n" +
                "Loss Trade %               : {fail_trades_pct}" + "\n" +
                "Profit Trade %             : {success_trades_pct}" + "\n" +
                "Largest Loss Trade         : {largest_loss_trade_value}" + "\n" +
                "Largest Profit Trade       : {largest_profit_trade_value}" + "\n" +
                "Consecutive Loss Trade     : {max_consecutive_lose_trades}" + "\n" +
                "Consecutive Profit Trade   : {max_consecutive_win_trades}" + "\n" +
                "--------------------{commodity}--------------------" + "\n"
        ).format(
            commodity=self.commodity,
            cum_ret=self.cumulative_return,
            tot_trades = self.total_trades,
            tot_succ_trades=self.total_success_trades,
            tot_fail_trad=self.total_fail_trades,
            avg_profit_trade=self.avg_profit_trade_value,
            avg_loss_trade=self.avg_loss_trade_value,
            fail_trades_pct=self.fail_trades_pct,
            success_trades_pct=self.success_trades_pct,
            largest_loss_trade_value=self.largest_loss_trade_value,
            largest_profit_trade_value=self.largest_profit_trade_value,
            max_consecutive_lose_trades=self.max_consecutive_lose_trades,
            max_consecutive_win_trades=self.max_consecutive_win_trades
        )
