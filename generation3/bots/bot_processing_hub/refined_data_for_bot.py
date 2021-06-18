from typing import Dict, List, Optional, Tuple, Union
import os.path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

from generation3.bots.bot_processing_hub.PCA import pcaUtility
from sys import platform as _platform


class RefinedData:
    """
    ticker: Name of the symbol to deliver
    features: The name of features required
    data_split_strategy: (% training set, % validation set, % test set)
    should_normalize: normalizes each column
    pca_strategy: if no -> {'use': false} if yes -> {'use': true, 'pca_components': number_of_components}
    """
    _ticker: str
    _features: Union[List[str], str]
    _data_split_strategy: Tuple[float, float, float]
    _should_normalize: bool
    _pca_strategy: Dict[str, any]
    _shuffle: bool
    _penalty_dataset: bool

    _predictors: pd.DataFrame
    _predictors_labels: List[str]
    _targets: pd.DataFrame
    _buy_sell_dataset: str

    # TrainX, TrainY
    # ValidationX, ValidationY
    # TestX, TestY
    _train_validate_test_set: Tuple[
        pd.DataFrame, pd.DataFrame,
        pd.DataFrame, pd.DataFrame,
        pd.DataFrame, pd.DataFrame
    ]

    def __init__(
            self,
            ticker: str,
            features: Union[List[str], str],
            data_split_strategy: Tuple[float, float, float],
            should_normalize: bool,
            pca_strategy: Dict[str, any],
            shuffle=False,
            penalty_dataset=False,
            buy_sell_dataset='b'
    ):
        self._ticker = ticker
        self._features = features
        self._data_split_strategy = data_split_strategy
        self._should_normalize = should_normalize
        self._pca_strategy = pca_strategy
        self._shuffle = shuffle
        self._penalty_dataset = penalty_dataset
        self._buy_sell_dataset = buy_sell_dataset

    @staticmethod
    def get_dataset_common(pca_components: Optional[int], should_use_penalty_dataset=False, buy_sell_dataset='b') -> Tuple[
        pd.DataFrame, pd.DataFrame,
        pd.DataFrame, pd.DataFrame,
        pd.DataFrame, pd.DataFrame
    ]:
        should_use_pca = pca_components is not None and pca_components is not 0
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
            shuffle=False,
            penalty_dataset=should_use_penalty_dataset,
            buy_sell_dataset=buy_sell_dataset
        ).get_data()


    def get_data(self) -> Tuple[
        pd.DataFrame, pd.DataFrame,
        pd.DataFrame, pd.DataFrame,
        pd.DataFrame, pd.DataFrame
    ]:
        self._get_predictors()
        self._get_targets()
        self._train_validation_test_splitter()
        self._normalize_individual_datasets()
        self._perform_pca()
        return self._train_validate_test_set

    def _get_predictors(self):

        # Assign Filepath

        if _platform == "darwin":
            file_path = '/Users/rahul/Main/CloudStation/Spizen/spizen-forex/master-insight-trading-bot-2/generation3/bots/datasets/' + self._ticker + '_input_signals.csv'
        else:
            file_path = '/student/ravin125/r/master-insight/generation3/bots/datasets/' + self._ticker + '_input_signals.csv'

        # Check if the ticker csv exists.
        if not os.path.exists(file_path):
            raise Exception("Predictor csv for the ticker is not available.")

        # Read predictors from csv
        self._predictors = pd.read_csv(file_path)

        # Remove irrelevant predictors
        if not self._features == "all":
            self._predictors = self._predictors[self._features]

        # Get predictor names
        self._predictors_labels = list(self._predictors.columns)

    def _get_targets(self):

        # Assign Filepath

        if self._penalty_dataset:
            if _platform == "darwin":
                file_path = '/Users/rahul/Main/CloudStation/Spizen/spizen-forex/master-insight-trading-bot-2/generation3/bots/datasets/' + self._ticker + '_clf_proxy_index.csv'
            else:
                file_path = '/student/ravin125/r/master-insight/generation3/bots/datasets/' + self._ticker + '_clf_proxy_index.csv'
        elif self._buy_sell_dataset == 'b':
            if _platform == "darwin":
                file_path = '/Users/rahul/Main/CloudStation/Spizen/spizen-forex/master-insight-trading-bot-2/generation3/bots/datasets/' + self._ticker + '_output_signals_buy.csv'
            else:
                file_path = '/student/ravin125/r/master-insight/generation3/bots/datasets/' + self._ticker + '_output_signals_buy.csv'
        else:
            if _platform == "darwin":
                file_path = '/Users/rahul/Main/CloudStation/Spizen/spizen-forex/master-insight-trading-bot-2/generation3/bots/datasets/' + self._ticker + '_output_signals_sell.csv'
            else:
                file_path = '/student/ravin125/r/master-insight/generation3/bots/datasets/' + self._ticker + '_output_signals_sell.csv'

        # Check if the ticker csv exists.
        if not os.path.exists(file_path):
            raise Exception("Target csv for this ticker is not available.")

        # Read target from csv
        self._targets = pd.read_csv(file_path)

    def _train_validation_test_splitter(self):

        train_ratio = self._data_split_strategy[0]
        validation_ratio = self._data_split_strategy[1]
        test_ratio = self._data_split_strategy[2]

        # train is now 75% of the entire data set
        # the _junk suffix means that we drop that variable completely
        x_train, x_test, y_train, y_test = train_test_split(
            self._predictors,
            self._targets,
            test_size=1 - train_ratio,
            shuffle=self._shuffle
        )

        # test is now 10% of the initial data set
        # validation is now 15% of the initial data set
        x_val, x_test, y_val, y_test = train_test_split(
            x_test,
            y_test,
            test_size=test_ratio / (test_ratio + validation_ratio),
            shuffle=self._shuffle
        )

        self._train_validate_test_set = (x_train, y_train, x_val, y_val, x_test, y_test)

    def _normalize_individual_datasets(self):
        if not self._should_normalize:
            return

        for dataset in self._train_validate_test_set:
            minmax_scale(
                dataset,
                feature_range=(0,1),
                axis=0,
                copy=False
            )

    def _perform_pca(self):

        if self._pca_strategy["use"] is False:
            return

        required_labels = list(
            set(self._predictors_labels) -
            set(self._pca_strategy["pca_features_to_exclude"])
        )

        new_datasets = []

        i = 0
        for dataset in self._train_validate_test_set:

            if i % 2 != 0:
                new_datasets.append(self._train_validate_test_set[i])
                i += 1
                continue

            new_datasets.append(
                pcaUtility.perform_pca(
                    required_labels,
                    dataset,
                    self._pca_strategy["pca_components"]
                )
            )
            i += 1

        self._train_validate_test_set = tuple(new_datasets)

if __name__ == "__main__":
    a = RefinedData(
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
    )

    a.get_data()

    print(a)
