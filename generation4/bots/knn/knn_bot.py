from sklearn.neighbors import KNeighborsClassifier

from generation4.bots.bot_template import TradingBot
import numpy as np


class KnnTradingBot(TradingBot):

    def predict(self) -> np.array:

        neigh = KNeighborsClassifier(
            n_neighbors=self._metadata['n_neighbors'],
            weights=self._metadata['weights']
        )

        neigh.fit(self._tr_x, self._tr_y)

        return neigh.predict_proba(self._va_x.to_numpy())[:, 1]
