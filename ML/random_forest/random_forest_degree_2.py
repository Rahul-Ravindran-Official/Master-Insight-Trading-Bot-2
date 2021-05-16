import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt


class RandomForestAlgorithmDegree2:

    def algorithm(self):

        # Read Dataset
        features = pd.read_csv('data_set_2/prediction.csv')
        labels = features["Actual"]
        del features["Actual"]

        # Split the data into training and testing sets
        train_features, test_features, train_labels, test_labels = train_test_split(
            features,
            labels,
            test_size=0.25,
            random_state=42,
            shuffle=False
        )

        train_labels = np.ravel(train_labels)
        test_labels = np.ravel(test_labels)

        rf = RandomForestRegressor(n_estimators=1000, random_state=42)
        rf.fit(train_features, train_labels)

        # Use the forest's predict method on the test data
        predictions = rf.predict(test_features)

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
            'Predictions': (predictions >= 0.5).astype(int),
            'Slope': dy,
            'Actual': test_labels
        })

        # manual_comparison = pd.DataFrame({
        #     'Predictions': predictions,
        #     'Slope': dy,
        #     'Actual': test_labels
        # })

        pd.DataFrame(manual_comparison).to_csv(
            r'/Users/rahul/Main/CloudStation/Spizen/spizen-forex/master-insight-trading-bot-2/ML/random_forest/prediction_deg_2.csv',
            index=False,
            header=True
        )

        # #### Visualization
        plt.plot(predictions, c="b")
        plt.plot(test_labels, c="g")
        plt.plot(dy, c="r")
        # plt.plot(features["4"][948:].to_numpy(), c="g")
        plt.show()


if __name__ == "__main__":
    RandomForestAlgorithmDegree2().algorithm()
