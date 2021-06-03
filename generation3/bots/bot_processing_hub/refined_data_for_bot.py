from typing import Dict, List, Tuple, Union
import os.path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

from generation3.bots.bot_processing_hub.PCA import pcaUtility


class RefinedDataForBot:
    """
    ticker: Name of the symbol to deliver
    features: The name of features required
    data_split_strategy: (% training set, % validation set, % test set)
    should_normalize: normalizes each column
    pca_strategy: if no -> {'use': false} if yes -> {'use': true, 'pca_components': number_of_components}
    """
    ticker: str
    features: Union[List[str], str]
    data_split_strategy: Tuple[float, float, float]
    should_normalize: bool
    pca_strategy: Dict[str, any]
    shuffle: bool

    predictors: pd.DataFrame
    predictors_labels: List[str]
    targets: pd.DataFrame

    # TrainX, TrainY
    # ValidationX, ValidationY
    # TestX, TestY
    train_validate_test_set: Tuple[
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
            shuffle=False
    ):
        self.ticker = ticker
        self.features = features
        self.data_split_strategy = data_split_strategy
        self.should_normalize = should_normalize
        self.pca_strategy = pca_strategy
        self.shuffle = shuffle

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
        return self.train_validate_test_set

    def _get_predictors(self):

        # Assign Filepath
        file_path = '../../' + self.ticker + '_input_signals.csv'

        # Check if the ticker csv exists.
        if not os.path.exists(file_path):
            raise Exception("Predictor csv for this ticker is not available.")

        # Read predictors from csv
        self.predictors = pd.read_csv(file_path)

        # Remove irrelevant predictors
        if not self.features == "all":
            self.predictors = self.predictors[self.features]

        # Get predictor names
        self.predictors_labels = list(self.predictors.columns)

    def _get_targets(self):

        # Assign Filepath
        file_path = '../../' + self.ticker + '_output_signals.csv'

        # Check if the ticker csv exists.
        if not os.path.exists(file_path):
            raise Exception("Target csv for this ticker is not available.")

        # Read target from csv
        self.targets = pd.read_csv(file_path)

    def _train_validation_test_splitter(self):

        train_ratio = self.data_split_strategy[0]
        validation_ratio = self.data_split_strategy[1]
        test_ratio = self.data_split_strategy[2]

        # train is now 75% of the entire data set
        # the _junk suffix means that we drop that variable completely
        x_train, x_test, y_train, y_test = train_test_split(
            self.predictors,
            self.targets,
            test_size=1 - train_ratio,
            shuffle=self.shuffle
        )

        # test is now 10% of the initial data set
        # validation is now 15% of the initial data set
        x_val, x_test, y_val, y_test = train_test_split(
            x_test,
            y_test,
            test_size=test_ratio / (test_ratio + validation_ratio),
            shuffle=self.shuffle
        )

        self.train_validate_test_set = (x_train, y_train, x_val, y_val, x_test, y_test)

    def _normalize_individual_datasets(self):
        if not self.should_normalize:
            return

        for dataset in self.train_validate_test_set:
            minmax_scale(
                dataset,
                feature_range=(0,1),
                axis=0,
                copy=False
            )

    def _perform_pca(self):

        if self.pca_strategy["use"] is False:
            return

        required_labels = list(
            set(self.predictors_labels) -
            set(self.pca_strategy["pca_features_to_exclude"])
        )

        new_datasets = []

        i = 0
        for dataset in self.train_validate_test_set:

            if i % 2 != 0:
                new_datasets.append(self.train_validate_test_set[i])
                i += 1
                continue

            new_datasets.append(
                pcaUtility.perform_pca(
                    required_labels,
                    dataset,
                    self.pca_strategy["pca_components"]
                )
            )
            i += 1

        self.train_validate_test_set = tuple(new_datasets)

if __name__ == "__main__":
    a = RefinedDataForBot(
        "NZDUSD=X",
        "all",
        (0.7, 0.15, 0.15),
        True,
        {
            'use': False,
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
