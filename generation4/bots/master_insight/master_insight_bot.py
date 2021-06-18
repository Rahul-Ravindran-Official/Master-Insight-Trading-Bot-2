from typing import Dict, List

from generation4.bot_dataset import BotDataset
from generation4.bots.bot_template import TradingBot
import numpy as np

from generation4.bots.knn.knn_bot import KnnTradingBot
from generation4.bots.svm.svm_bot import SVMTradingBot
import pandas as pd


class MasterInsightBot(TradingBot):

    _bots: List[TradingBot]

    def __init__(self, tr_x: pd.DataFrame, tr_y: pd.DataFrame, va_x: pd.DataFrame, metadata: Dict):
        super().__init__(tr_x, tr_y, va_x, metadata)
        self._bots = self.get_bots()

    def get_bots(self) -> List[TradingBot]:

        bot_types: List[type(TradingBot)] = [KnnTradingBot, SVMTradingBot]

        bots: List[TradingBot] = []

        for bot_type in bot_types:
            bots.append(
                bot_type(
                    self._tr_x,
                    self._tr_y,
                    self._va_x,
                    self._metadata['bots'][bot_type]
                )
            )

        return bots

    def predict(self) -> np.array:

        prediction_matrix = None

        for bot in self._bots:
            prediction = bot.predict()
            if prediction_matrix is None:
                prediction_matrix = prediction
            else:
                prediction_matrix = np.vstack((prediction_matrix, prediction))

        return prediction_matrix

def get_bot_dataset():
    dataset = BotDataset(
        pip_definition=0.0001,
        take_profit_pips=100,
        stop_loss_pips=10,
        spread=0,
        maximum_days_to_hold_per_trade=5
    ).return_data()

    b_tr_x = dataset['b_tr_x']
    b_tr_y = dataset['b_tr_y']
    b_va_x = dataset['b_va_x']

    return b_tr_x, b_tr_y, b_va_x


if __name__ == '__main__':

    b_tr_x, b_tr_y, b_va_x = get_bot_dataset()

    p = MasterInsightBot(
        b_tr_x, b_tr_y, b_va_x,
        {
            'bots': {
                KnnTradingBot: {
                    'n_neighbors': 3,
                    'weights': 'distance'
                },
                SVMTradingBot: {
                    'kernel': 'rbf'
                }
            }
        }
    ).predict()

    res = 9
