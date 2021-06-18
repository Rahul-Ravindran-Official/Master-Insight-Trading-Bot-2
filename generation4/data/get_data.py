from typing import Dict, List
import talib
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.preprocessing import minmax_scale


class ForexData:

    ohlc: pd.DataFrame
    ohlc_indicators: pd.DataFrame

    # 45K Records
    train_data: pd.DataFrame
    # 6 Datasets -> 30K Records -> 5K Records each
    validate_data: List[pd.DataFrame]
    # 3 Datasets -> 15K Records -> 5K Records each
    test_data: List[pd.DataFrame]

    def __init__(self):
        self.ohlc = pd.DataFrame([])

    def get_data(self):
        self.ohlc = pd.read_csv('/Users/rahul/Main/CloudStation/Spizen/spizen-forex/master-insight-trading-bot-2/generation4/data/NZDUSD_H1.csv', sep="\t")

    def drop_na(self):
        self.ohlc.dropna()

    def dataset_normalization(self, dataset: pd.DataFrame):
        minmax_scale(
            self.ohlc_indicators,
            feature_range=(0, 1),
            axis=0,
            copy=False
        )

        minmax_scale(
            dataset['Volume'],
            feature_range=(0, 1),
            axis=0,
            copy=False
        )

        minmax_scale(
            dataset['day'],
            feature_range=(0, 1),
            axis=0,
            copy=False
        )

    def feature_engineering(self):

        # Add Cyclic Time
        temp = self.ohlc_indicators['Time'].str.split(' ', expand=True)
        self.ohlc_indicators['Date'], self.ohlc_indicators['Time'] = temp[0], temp[1]
        self.ohlc_indicators['Time'] = self.ohlc_indicators['Time'].str.split(':', expand=True)[0]
        self.ohlc_indicators['hour_sin'] = np.sin(2 * np.pi * self.ohlc_indicators['Time'].astype(dtype=float) / 23.0)
        self.ohlc_indicators['hour_cos'] = np.cos(2 * np.pi * self.ohlc_indicators['Time'].astype(dtype=float) / 23.0)

        # Add Cyclic Day
        days = []
        for date in self.ohlc_indicators['Date']:
            days.append(datetime.strptime(date, '%Y-%m-%d').weekday())

        self.ohlc_indicators['day'] = days

        self.ohlc.drop(labels=['Date', 'Time'], axis=1, inplace=True)

    def split_datasets_and_normalize(self):

        self.train_data = self.ohlc_indicators.iloc[0:45000, :]
        self.train_data.reset_index(drop=True, inplace=True)
        self.dataset_normalization(self.train_data)

        self.validate_data = [
            self.ohlc_indicators.iloc[45001:50001, :],
            self.ohlc_indicators.iloc[50001:55001, :],
            self.ohlc_indicators.iloc[55001:60001, :],
            self.ohlc_indicators.iloc[60001:65001, :],
            self.ohlc_indicators.iloc[65001:70001, :],
            self.ohlc_indicators.iloc[70001:75001, :]
        ]

        for dataset in self.validate_data:
            dataset.reset_index(drop=True, inplace=True)
            self.dataset_normalization(dataset)

        self.test_data = [
            self.ohlc_indicators.iloc[75001:80001, :],
            self.ohlc_indicators.iloc[80001:85001, :],
            self.ohlc_indicators.iloc[85001:90001, :]
        ]

        for dataset in self.test_data:
            dataset.reset_index(drop=True, inplace=True)
            self.dataset_normalization(dataset)


    def obtain(self):
        self.get_data()
        self.ohlc_indicators = self.all_technical_indicator_inputs(self.ohlc)
        self.feature_engineering()
        self.drop_na()
        self.split_datasets_and_normalize()
        return self.train_data, self.validate_data, self.test_data

    @staticmethod
    def all_technical_indicator_inputs(ohlc_data: pd.DataFrame) -> pd.DataFrame:

        inputs = {
            'open': ohlc_data['Open'],
            'high': ohlc_data['High'],
            'low': ohlc_data['Low'],
            'close': ohlc_data['Close'],
            'volume': ohlc_data['Volume']
        }

        # Price Data
        def day_diff(diff_days: int):
            n_day_diff = ohlc_data["Close"][diff_days:].to_numpy() - ohlc_data["Close"][:-diff_days].to_numpy()
            n_day_diff = np.insert(n_day_diff, 0, [np.nan] * diff_days, axis=0)
            return n_day_diff

        for i in range(5):
            k = i + 1
            ohlc_data["diff-day-" + str(k)] = day_diff(k)

        # Simple Indicators
        ohlc_data["EMA-2"] = ohlc_data["Close"] - talib.EMA(
            ohlc_data["Close"], 2)
        ohlc_data["EMA-3"] = ohlc_data["Close"] - talib.EMA(
            ohlc_data["Close"], 3)
        ohlc_data["EMA-5"] = ohlc_data["Close"] - talib.EMA(
            ohlc_data["Close"], 5)
        ohlc_data["EMA-7"] = ohlc_data["Close"] - talib.EMA(
            ohlc_data["Close"], 7)
        ohlc_data["EMA-21"] = ohlc_data["Close"] - talib.EMA(
            ohlc_data["Close"], 21)
        ohlc_data["EMA-50"] = ohlc_data["Close"] - talib.EMA(
            ohlc_data["Close"], 50)
        ohlc_data["RSI"] = talib.RSI(ohlc_data["Close"], 14) / 100
        ohlc_data["MACD"] = talib.MACD(ohlc_data["Close"], 12, 26, 9)[2]
        ohlc_data["ADX"] = talib.ADX(ohlc_data["High"], ohlc_data["Low"],
                                     ohlc_data["Close"]) / 100
        ohlc_data["STD"] = talib.STDDEV(ohlc_data["Close"])

        # Raw Indicators
        ohlc_data["RAW-EMA-2"] = talib.EMA(ohlc_data["Close"], 2)
        ohlc_data["RAW-EMA-3"] = talib.EMA(ohlc_data["Close"], 3)
        ohlc_data["RAW-EMA-5"] = talib.EMA(ohlc_data["Close"], 5)
        ohlc_data["RAW-EMA-7"] = talib.EMA(ohlc_data["Close"], 7)
        ohlc_data["RAW-EMA-21"] = talib.EMA(ohlc_data["Close"], 21)
        ohlc_data["RAW-EMA-50"] = talib.EMA(ohlc_data["Close"], 50)

        # Complex Indicators
        def get_useful_bbands():
            bb = talib.BBANDS(ohlc_data["Close"])
            bb_width = bb[0] - bb[2]
            bb_buy = ohlc_data["Close"] - bb[0]
            bb_sell = bb[2] - ohlc_data["Close"]
            return bb_width, bb_buy, bb_sell

        ohlc_data["bb_width"], ohlc_data["bb_buy"], ohlc_data[
            "bb_sell"] = get_useful_bbands()

        ohlc_data['Volume'] = ohlc_data['Volume'].diff()

        return ohlc_data

if __name__ == '__main__':
    d = ForexData().obtain()
