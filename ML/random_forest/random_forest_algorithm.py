import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier


class RandomForestAlgorithm:

    def algorithm(self):

        # Read Dataset
        features = pd.read_csv('data_set_forex_simple_n/A.csv')
        feature_list = list(features.columns)
        labels = pd.read_csv('data_set_forex_simple_n/b.csv')

        goodness_values = pd.read_csv('data_set_forex_simple_n/value.csv')

        # Split the data into training and testing sets -> Note change to this!
        train_features, test_features, train_labels, test_labels = train_test_split(
            features,
            labels,
            test_size=0.50,
            random_state=42,
            shuffle=False
        )

        # train_features, test_features, train_labels, test_labels = train_test_split(
        #     features,
        #     labels,
        #     test_size=0.25,
        #     random_state=42,
        #     shuffle=True
        # )

        train_labels = np.ravel(train_labels)
        test_labels = np.ravel(test_labels)


        # # Forrest Algo
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(train_features, train_labels)

        # Use the forest's predict method on the test data
        predictions = rf.predict(test_features)



        # mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000, activation='relu', verbose=True)
        # mlp.fit(train_features, train_labels)
        # predictions = mlp.predict(test_features)

        # Normalise Predictions if required
        predictions_norm = minmax_scale(
            predictions[:],
            feature_range=(min(test_labels), max(test_labels)),
            axis=0,
            copy=True
        )

        test_labels_norm = minmax_scale(
            test_labels[:],
            feature_range=(-10, 10),
            axis=0,
            copy=True
        )

        # Calculate the absolute errors
        errors = abs(predictions - test_labels)

        # Print out the mean absolute error (mae)
        print('Mean Absolute Error:', round(np.mean(errors), 2), 'Points.')

        # Calculate the slope
        dy = np.diff(predictions_norm)
        dy = list(dy)
        dy.insert(0, dy[1])
        dy = np.array(dy)

        manual_comparison = pd.DataFrame({
            'Predictions': predictions,
            'Slope': dy,
            'Actual': test_labels,
            'Goodness': np.array(goodness_values[:len(test_labels)]).reshape(len(test_labels),)
        })


        manual_comparison = pd.DataFrame({
            'Predictions': (predictions >= 0.5).astype(int),
            'Slope': dy,
            'Actual': test_labels,
            'Goodness': np.array(goodness_values[:len(test_labels)]).reshape(len(test_labels),)
        })


        pd.DataFrame(manual_comparison).to_csv(
            r'/Users/rahul/Main/CloudStation/Spizen/spizen-forex/master-insight-trading-bot-2/ML/random_forest/data_set_forex_simple_n/prediction.csv',
            index=False,
            header=True
        )

        signal = (predictions >= 0.5).astype(int)

        pd.DataFrame(signal).to_csv(
            r'/Users/rahul/Main/CloudStation/Spizen/spizen-forex/master-insight-trading-bot-2/ML/random_forest/data_set_forex_simple_n/buy_signal.csv',
            index=False,
            header=True
        )

        # #### Visualization
        plt.plot(predictions, c="b")
        plt.plot(test_labels, c="g")
        plt.plot(dy, c="r")
        # plt.plot(features["4"][948:].to_numpy(), c="g")
        plt.show()

        importance_data = ""

        # Get numerical feature importances
        importances = list(rf.feature_importances_)
        # List of tuples with variable and importance
        feature_importances = [(feature, round(importance, 2)) for
                               feature, importance in
                               zip(feature_list, importances)]
        # Sort the feature importances by most important first
        feature_importances = sorted(feature_importances, key=lambda x: x[1],
                                     reverse=True)
        # Print out the feature and importances
        importance_data = ['Variable: {:20} Importance: {}'.format(*pair) for pair in
         feature_importances]

        print(importance_data)


if __name__ == "__main__":
    RandomForestAlgorithm().algorithm()
