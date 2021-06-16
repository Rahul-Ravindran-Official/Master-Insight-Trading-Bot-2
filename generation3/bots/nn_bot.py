import unittest
from typing import Dict

from keras import models
from keras import layers

import pandas as pd
import numpy as np

from generation3.bots.bot_processing_hub.backtester.BotBackTester import \
    BotBackTester
from generation3.bots.bot_processing_hub.evaluators.loss_function import penalty_loss_function
from generation3.bots.bot_processing_hub.refined_data_for_bot import RefinedData
from generation3.ensemble_trading_bot import SignalProvider


class NNBot(SignalProvider):

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
        return NNBot_V1(
            self._b_t_x, self._b_t_y, self._b_v_x,
            self._s_t_x, self._s_t_y, self._s_v_x,
            self._metadata
        ).get_buy_signal()

    def get_sell_signal(self) -> np.array:
        return NNBot_V1(
            self._b_t_x, self._b_t_y, self._b_v_x,
            self._s_t_x, self._s_t_y, self._s_v_x,
            self._metadata
        ).get_sell_signal()

    @staticmethod
    def argument_nn_bot(args: Dict):

        b_t_x, b_t_y, b_v_x, _, _, _ = RefinedData.get_dataset_common(
            args['pca']['pca_components_buy'],
            buy_sell_dataset='b'
        )

        s_t_x, s_t_y, s_v_x, _, _, _ = RefinedData.get_dataset_common(
            args['pca']['pca_components_sell'],
            buy_sell_dataset='s'
        )

        bbt = BotBackTester(
            NNBot(
                b_t_x, b_t_y, b_v_x,
                s_t_x, s_t_y, s_v_x,
                args
            ),
            list(b_v_x["Adj Close"]),
            buy_treshold=(
                args['type']['buy_signal']['treshold']['low'],
                args['type']['buy_signal']['treshold']['high']
            ),
            sell_treshold=(
                args['type']['sell_signal']['treshold']['low'],
                args['type']['sell_signal']['treshold']['high']
            ),
            spread=0,
            lot_size=0.1,
            pip_definition=0.0001,
            profit_per_pip_per_lot=10
        )

        bbt.back_test()
        return bbt.get_stats()

    @staticmethod
    def hyperopt_nn_objective(args: Dict):
        return -NNBot.argument_nn_bot(args)["profits_earned"]


class NNBot_V1(SignalProvider):

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

    def get_buy_signal(self) -> np.array:
        return self.generic_signal_producer(
            self._b_t_x, self._b_t_y, self._b_v_x,
            self._metadata['type']['buy_signal']
        )

    def get_sell_signal(self) -> np.array:
        return self.generic_signal_producer(
            self._s_t_x, self._s_t_y, self._s_v_x,
            self._metadata['type']['sell_signal']
        )

    @staticmethod
    def generic_signal_producer(
            t_x, t_y, v_x,
            args: Dict
    ) -> np.array:

        model = models.Sequential()

        model.add(layers.Dense(args['nn']['l1-nodes'], activation='relu', input_shape=(t_x.shape[1],)))
        model.add(layers.Dropout(args['nn']['l1-dropout']))

        model.add(layers.Dense(args['nn']['l2-nodes'], activation='relu'))
        model.add(layers.Dropout(args['nn']['l2-dropout']))

        model.add(layers.Dense(args['nn']['l3-nodes'], activation='relu'))
        model.add(layers.Dropout(args['nn']['l3-dropout']))

        model.add(layers.Dense(args['nn']['l4-nodes'], activation='relu'))
        model.add(layers.Dropout(args['nn']['l4-dropout']))

        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(
            optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        history = model.fit(
            t_x,
            t_y,
            epochs=args['epochs'],
            batch_size=512,
            validation_split=0.2,
            class_weight={
                '0': args['nn']['class_weight']['class-weight-0'],
                '1': args['nn']['class_weight']['class-weight-1']
            }
        )

        return model.predict(v_x)



class NNBotPlayground(unittest.TestCase):

    @staticmethod
    def test_nn_generic():

        train_x, train_y, validation_x, validation_y, _, _ = RefinedData.get_dataset_common(None)

        model = models.Sequential()
        model.add(layers.Dense(5, activation='relu', input_shape=(train_x.shape[1],)))
        model.add(layers.Dense(5, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(
            optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        history = model.fit(
            train_x,
            train_y,
            epochs=1,
            batch_size=512,
            validation_data=(validation_x, validation_y)
        )

        assert 1 == 1

    @staticmethod
    def test_nn_normalised_custom_penalty():

        train_x, train_y, validation_x, validation_y, _, _ = RefinedData.get_dataset_common(None, True)

        model = models.Sequential()
        model.add(layers.Dense(5, activation='relu', input_shape=(train_x.shape[1],)))
        model.add(layers.Dense(5, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(
            optimizer='rmsprop',
            loss=penalty_loss_function,
            metrics=['accuracy']
        )

        history = model.fit(
            train_x,
            train_y,
            epochs=5,
            batch_size=128,
            validation_data=(validation_x, validation_y)
        )

        assert 1 == 1
