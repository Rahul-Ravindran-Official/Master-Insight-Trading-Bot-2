from sklearn import svm
from generation4.bots.bot_template import TradingBot
import numpy as np


class SVMTradingBot(TradingBot):

    def predict(self) -> np.array:

        clf = svm.SVC(
            class_weight='balanced',
            kernel=self._metadata['kernel'],
            probability=True
        )

        clf.fit(self._tr_x, self._tr_y)

        return clf.predict_proba(self._va_x.to_numpy())[:, 1]

