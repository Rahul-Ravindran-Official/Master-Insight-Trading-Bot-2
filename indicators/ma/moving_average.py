import pandas as pd

from indicators.ma.ma_type import MAType


class MovingAverage:
    period: int
    type: MAType

    def __init__(self, period: int = 12, type: MAType = MAType.ema):
        self.period = period
        self.type = type

    def get_signal(
            self,
            ohlc_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculates and returns moving average

        :param ohlc_df: A yfinance OHLC Dataframe
        :return: Dataframe with columns "ma_period_<the-period>"
        """
        df = ohlc_df.copy()

        # Intermediary Computations - Stage 1
        if self.type == MAType.ema:
            df["ma_period_" + str(self.period)] = df["Adj Close"].ewm(
                span=self.period,
                min_periods=self.period
            ).mean()
        else:
            df["ma_period_" + str(self.period)] = df["Adj Close"].rolling(
                window=self.period
            ).mean()

        df.dropna(inplace=True)

        return df[["ma_period_" + str(self.period)]]
