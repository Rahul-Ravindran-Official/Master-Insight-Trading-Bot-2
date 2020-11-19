from typing import Dict

import pandas as pd

from indicators.ma.ma_type import MAType
from indicators.ma.moving_average import MovingAverage
from shared.BBHSSSignal import BBHSSSignal
from shared.Strategy import Strategy


class MADoubleCrossOver(Strategy):
    period_fast: int
    period_slow: int

    def __init__(self, period_fast: int = 50, period_slow: int = 200):
        self.period_fast = period_fast
        self.period_slow = period_slow

    def get_signals(self, ohlc_df: pd.DataFrame) -> pd.DataFrame:
        working_df = ohlc_df.__deepcopy__()

        working_df, cols_ma_fast = MovingAverage(
            self.period_fast, MAType.ema
        ).get_signal(working_df)

        working_df, cols_ma_slow = MovingAverage(
            self.period_slow, MAType.ema
        ).get_signal(working_df)

        metadata = {
            'ma_fast': cols_ma_fast[0],
            'ma_slow': cols_ma_slow[0]
        }

        return self.process_to_bbhss(working_df, metadata)

    def process_to_bbhss(self, signals_and_ohlc_df: pd.DataFrame, metadata: Dict) -> pd.DataFrame:

        bbhss_mapper = {
            True: BBHSSSignal.buy,
            False: BBHSSSignal.sell
        }

        signals_and_ohlc_df.dropna(inplace=True)
        signals_and_ohlc_df['MDCO_SIG'] = signals_and_ohlc_df[metadata['ma_fast']] > signals_and_ohlc_df[metadata['ma_slow']]
        signals_and_ohlc_df['MDCO_SIG'] = signals_and_ohlc_df['MDCO_SIG'].map(bbhss_mapper)
        return signals_and_ohlc_df
