import unittest
from typing import List, Optional, Tuple

import numpy as np
from sklearn import svm
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import math

from generation3.bots.bot_processing_hub.refined_data_for_bot import \
    RefinedData
from generation3.ensemble_trading_bot import SignalProvider


class SVMBot(SignalProvider):
    def get_signal(self) -> np.array:
        return SVMBot_V1().get_signal()


class SVMBot_V1(SignalProvider):

    def get_dataset(self) -> Tuple[
        pd.DataFrame, pd.DataFrame,
        pd.DataFrame, pd.DataFrame,
        pd.DataFrame, pd.DataFrame
    ]:
        return RefinedData(
            "NZDUSD=X",
            "all",
            (0.7, 0.15, 0.15),
            True,
            {
                'use': True,
                'pca_components': 3,
                'pca_features_to_exclude': [
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Adj Close",
                    "Volume"
                ]
            },
            shuffle=False
        ).get_data()

    def get_signal(self) -> np.array:

        train_x, train_y, validation_x, validation_y, test_x, test_y = self.get_dataset()

        model_linear = svm.SVC(
            kernel='linear', class_weight={0: 1, 1: 1}
        )

        clf = model_linear.fit(train_x, train_y)

        pred_validation_y = clf.predict(validation_x)

        print(confusion_matrix(validation_y, pred_validation_y))


class SVMBotPlayground(unittest.TestCase):

    def get_dataset(self, pca_components: Optional[int]) -> Tuple[
        pd.DataFrame, pd.DataFrame,
        pd.DataFrame, pd.DataFrame,
        pd.DataFrame, pd.DataFrame
    ]:
        should_use_pca = pca_components is not None
        return RefinedData(
            "NZDUSD=X",
            "all",
            (0.7, 0.15, 0.15),
            True,
            {
                'use': should_use_pca,
                'pca_components': pca_components,
                'pca_features_to_exclude': [
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Adj Close",
                    "Volume"
                ]
            },
            shuffle=False
        ).get_data()

    def test_sanity(self):
        self.assertEqual(True, False)

    def test_svc_signal_linear_custom_weights(self):
        train_x, train_y, validation_x, validation_y, _, _ = self.get_dataset()

        model_linear = svm.SVC(
            kernel='linear',
            class_weight={0: 1, 1: 1}
        )

        clf = model_linear.fit(train_x, train_y)

        pred_validation_y = clf.predict(validation_x)

        print(confusion_matrix(validation_y, pred_validation_y))

        # Visualization
        plot_confusion_matrix(clf, validation_x, validation_y)
        plt.show()

    def test_svc_signal_linear_balanced(self):
        train_x, train_y, validation_x, validation_y, _, _ = self.get_dataset()

        model_linear = svm.SVC(
            kernel='linear',
            class_weight='balanced'
        )

        clf = model_linear.fit(train_x, train_y)

        pred_validation_y = clf.predict(validation_x)

        print(confusion_matrix(validation_y, pred_validation_y))

        # Visualization
        plot_confusion_matrix(clf, validation_x, validation_y)
        plt.show()

    def test_svc_signal_rbf_balanced(self):
        train_x, train_y, validation_x, validation_y, _, _ = self.get_dataset()

        model_linear = svm.SVC(
            kernel='poly',
            degree=3,
            class_weight='balanced'
        )

        clf = model_linear.fit(train_x, train_y)

        pred_validation_y = clf.predict(validation_x)

        print(confusion_matrix(validation_y, pred_validation_y))

        # Visualization
        plot_confusion_matrix(clf, validation_x, validation_y)
        plt.show()


    def svc_cost_function(self, false_positives, true_positives):
        penalty_of_false_positive = -false_positives
        incentive_for_predicting_positive = math.sqrt(false_positives + true_positives)
        # return math.pow(true_positives, 2) + penalty_of_false_positive + incentive_for_predicting_positive
        return (true_positives/false_positives)

    def test_svc_search_space(self):

        grid_search_params = {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'pca_components': [1, 2, 3, 4, 5, None]
        }

        best_args = {
            'kernel': '',
            'pca_components': '',
            'score': -float('inf')
        }

        i = 0

        for pca_component in grid_search_params["pca_components"]:
            for kernel in grid_search_params["kernel"]:

                if kernel == "poly" and pca_component == 1:
                    continue

                train_x, train_y, validation_x, validation_y, _, _ = self.get_dataset(
                    pca_component
                )

                model_linear = svm.SVC(
                    kernel=kernel,
                    class_weight='balanced'
                )

                clf = model_linear.fit(train_x, train_y)

                pred_validation_y = clf.predict(validation_x)

                cm = confusion_matrix(validation_y, pred_validation_y)

                false_positives = cm[0][1]
                true_positives = cm[1][1]

                cost_function_score = self.svc_cost_function(false_positives, true_positives)

                print("----------")
                print("Iteration : " + str(i))
                print("Config: " + "kernel - " + kernel + " :: pca_components : " + str(pca_component))
                print("Score : " + str(cost_function_score))
                print(cm)
                print("----------")

                if best_args['score'] < cost_function_score:
                    best_args['score'] = cost_function_score
                    best_args['kernel'] = kernel
                    best_args['pca_components'] = pca_component

                i += 1

        print(best_args)
