from typing import Dict

from generation3.bots.bot_processing_hub.backtester.BotBackTester import \
    BotBackTester
from generation3.bots.bot_processing_hub.refined_data_for_bot import RefinedData
from generation3.ensemble_trading_bot import SignalProvider
import numpy as np


class TheoreticalBot(SignalProvider):

    buy_signal: np.array
    sell_signal: np.array

    def __init__(
            self,
            b_t_x, b_t_y, b_v_x,
            s_t_x, s_t_y, s_v_x,
            metadata: Dict
    ):

        self.buy_signal = metadata["true_buy_signal"]
        self.sell_signal = metadata["true_sell_signal"]

        self.buy_signal = RefinedData.get_dataset_common(
            None,
            buy_sell_dataset='b'
        )[3].to_numpy()

        self.buy_signal = self.buy_signal.reshape(len(self.buy_signal), )

        self.sell_signal = RefinedData.get_dataset_common(
            None,
            buy_sell_dataset='s'
        )[3].to_numpy()

        self.sell_signal = self.sell_signal.reshape(len(self.sell_signal), )

    def get_buy_signal(self) -> np.array:
        return self.buy_signal

    def get_sell_signal(self) -> np.array:
        return self.sell_signal


if __name__ == "__main__":

    b_t_x, b_t_y, b_v_x, true_buy_signal, _, _ = RefinedData.get_dataset_common(
        None,
        buy_sell_dataset='b'
    )

    s_t_x, s_t_y, s_v_x, true_sell_signal, _, _ = RefinedData.get_dataset_common(
        None,
        buy_sell_dataset='s'
    )

    bbt = BotBackTester(
        TheoreticalBot(
            None, None, None,
            None, None, None,
            {
                'true_buy_signal': true_buy_signal,
                'true_sell_signal': true_sell_signal
            }
        ),
        list(b_v_x["Adj Close"]),
        buy_treshold=(0.5, 1.0),
        sell_treshold=(0.9, 1.0),
        spread=0,
        lot_size=0.1,
        pip_definition=0.0001,
        profit_per_pip_per_lot=10
    )

    bbt.back_test()
    bbt.print_stats()
