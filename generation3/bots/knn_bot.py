from typing import Dict

from sklearn.neighbors import KNeighborsClassifier
import unittest

from generation3.bots.bot_processing_hub.backtester.BotBackTester import \
    BotBackTester
from generation3.bots.bot_processing_hub.refined_data_for_bot import RefinedData
import pandas as pd
import numpy as np

from generation3.ensemble_trading_bot import SignalProvider

class KNNBot(SignalProvider):

    _b_t_x: pd.DataFrame
    _b_t_y: pd.DataFrame
    _b_v_x: pd.DataFrame

    _s_t_x: pd.DataFrame
    _s_t_y: pd.DataFrame
    _s_v_x: pd.DataFrame
    _metadata: Dict

    def __init__(
            self,
            b_t_x, b_t_y, b_v_x,
            s_t_x, s_t_y, s_v_x,
            metadata: Dict
    ):
        self._b_t_x = b_t_x
        self._b_t_y = b_t_y
        self._b_v_x = b_v_x
        self._s_t_x = s_t_x
        self._s_t_y = s_t_y
        self._s_v_x = s_v_x
        self._metadata = metadata

    def get_buy_signal(self) -> np.array:
        return KNNBot_V1(
            self._b_t_x, self._b_t_y, self._b_v_x,
            self._s_t_x, self._s_t_y, self._s_v_x,
            self._metadata
        ).get_buy_signal()

    def get_sell_signal(self) -> np.array:
        return KNNBot_V1(
            self._b_t_x, self._b_t_y, self._b_v_x,
            self._s_t_x, self._s_t_y, self._s_v_x,
            self._metadata
        ).get_sell_signal()

class KNNBot_V1(SignalProvider):

    _b_t_x: pd.DataFrame
    _b_t_y: pd.DataFrame
    _b_v_x: pd.DataFrame

    _s_t_x: pd.DataFrame
    _s_t_y: pd.DataFrame
    _s_v_x: pd.DataFrame
    _neighbours: int

    def __init__(
            self,
            b_t_x, b_t_y, b_v_x,
            s_t_x, s_t_y, s_v_x,
            metadata: Dict
    ):
        self._b_t_x = b_t_x
        self._b_t_y = b_t_y
        self._b_v_x = b_v_x
        self._s_t_x = s_t_x
        self._s_t_y = s_t_y
        self._s_v_x = s_v_x
        self._neighbours = metadata['neighbours']

    def get_buy_signal(self) -> np.array:
        return self.generic_signal_producer(self._b_t_x, self._b_t_y, self._b_v_x)

    def get_sell_signal(self) -> np.array:
        return self.generic_signal_producer(self._s_t_x, self._s_t_y, self._s_v_x)

    def generic_signal_producer(self, t_x, t_y, v_x) -> np.array:
        neigh = KNeighborsClassifier(
            n_neighbors=self._neighbours
        )
        neigh.fit(t_x, t_y)
        return neigh.predict_proba(v_x.to_numpy())[:, 1]

class KNNBotPlayground():

    @staticmethod
    def test_knn_buy_signal():

        t_x, t_y, v_x, v_y, _, _ = RefinedData.get_dataset_common(
            None
        )

        t_x = t_x.to_numpy()
        t_y = t_y.to_numpy()

        neigh = KNeighborsClassifier(n_neighbors=1)
        neigh.fit(t_x, t_y)

        pred_df = neigh.predict_proba(v_x.to_numpy())

        manual_verification = pd.DataFrame({
            "Probability Buy": pred_df[:, 1],
            "Suggested Buy": neigh.predict(v_x.to_numpy()),
            "Real Buy": v_y.to_numpy().reshape((len(v_y),))
        })

        print(1)
        print(
            neigh.predict(
                [
                    v_x[0]
                ]
            )
        )


def knn_func(args: Dict):
    b_t_x, b_t_y, b_v_x, _, _, _ = RefinedData.get_dataset_common(
        None,
        buy_sell_dataset='b'
    )

    s_t_x, s_t_y, s_v_x, _, _, _ = RefinedData.get_dataset_common(
        None,
        buy_sell_dataset='s'
    )

    bbt = BotBackTester(
        KNNBot(
            b_t_x, b_t_y, b_v_x,
            s_t_x, s_t_y, s_v_x,
            {
                'neighbours': args['neighbours']
            }
        ),
        list(b_v_x["Adj Close"]),
        buy_treshold=(
        args['buy_treshold']['low'], args['buy_treshold']['high']),
        sell_treshold=(
        args['sell_treshold']['low'], args['sell_treshold']['high']),
        spread=0,
        lot_size=0.1,
        pip_definition=0.0001,
        profit_per_pip_per_lot=10
    )

    bbt.back_test()
    return bbt.get_stats()

def hyperopt_knn_objective(args: Dict):
    return -knn_func(args)["profits_earned"]

if __name__ == "__main__":

    # space = hp.choice('classifier_type', [
    #     {
    #         'type': 'naive_bayes',
    #     },
    #     {
    #         'type': 'svm',
    #         'C': hp.lognormal('svm_C', 0, 1),
    #         'kernel': hp.choice('svm_kernel', [
    #             {'ktype': 'linear'},
    #             {'ktype': 'RBF', 'width': hp.lognormal('svm_rbf_width', 0, 1)},
    #         ]),
    #     },
    #     {
    #         'type': 'dtree',
    #         'criterion': hp.choice('dtree_criterion', ['gini', 'entropy']),
    #         'max_depth': hp.choice('dtree_max_depth', [None, hp.qlognormal('dtree_max_depth_int', 3, 1, 1)]),
    #         'min_samples_split': hp.qlognormal('dtree_min_samples_split', 2, 1, 1),
    #     },
    # ])
    #
    # best = fmin(objective, space, algo=tpe.suggest, max_evals=100)

    b_t_x, b_t_y, b_v_x, _, _, _ = RefinedData.get_dataset_common(
        None,
        buy_sell_dataset='b'
    )

    s_t_x, s_t_y, s_v_x, _, _, _ = RefinedData.get_dataset_common(
        None,
        buy_sell_dataset='s'
    )

    bbt = BotBackTester(
        KNNBot(
            b_t_x, b_t_y, b_v_x,
            s_t_x, s_t_y, s_v_x,
            {'neighbours': 5}
        ),
        list(b_v_x["Adj Close"]),
        buy_treshold=(0.5, 1.0),
        sell_treshold=(0.1, 1.0),
        spread=0,
        lot_size=0.1,
        pip_definition=0.0001,
        profit_per_pip_per_lot=10
    )

    bbt.back_test()
    bbt.print_stats()

