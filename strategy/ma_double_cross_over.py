from typing import Dict, Tuple

import pandas as pd

from indicators.ma.ma_type import MAType
from indicators.ma.moving_average import MovingAverage
from shared.BBHSSSignal import BBHSSSignal
from shared.Strategy import Strategy


class MADoubleCrossOver(Strategy):
    period_fast: int
    period_slow: int
    signal_col_name: str

    def __init__(self, magic_no: int, period_fast: int = 50, period_slow: int = 200):
        self.period_fast = period_fast
        self.period_slow = period_slow
        self.signal_col_name = 'MADCO-' + str(magic_no)

    def get_bbhss_signal(self, ohlc_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        working_df = ohlc_df.__deepcopy__()

        working_df, cols_ma_fast = MovingAverage(
            ohlc_df=working_df,
            period=self.period_fast,
            ma_type=MAType.ema
        ).get_signal()

        working_df, cols_ma_slow = MovingAverage(
            ohlc_df=working_df,
            period=self.period_slow,
            ma_type=MAType.ema
        ).get_signal()

        metadata = {
            'ma_fast': cols_ma_fast[0],
            'ma_slow': cols_ma_slow[0]
        }

        return self.process_to_bbhss(working_df, metadata), self.signal_col_name

    def process_to_bbhss(self, signals_and_ohlc_df: pd.DataFrame, metadata: Dict) -> pd.DataFrame:

        bbhss_mapper = {
            True: BBHSSSignal.buy.value,
            False: BBHSSSignal.sell.value
        }

        signals_and_ohlc_df[self.signal_col_name] = signals_and_ohlc_df[metadata['ma_fast']] > signals_and_ohlc_df[metadata['ma_slow']]
        signals_and_ohlc_df[self.signal_col_name] = signals_and_ohlc_df[self.signal_col_name].map(bbhss_mapper)

        signals_and_ohlc_df.drop([metadata['ma_fast'], metadata['ma_slow']], axis=1, inplace=True)

        return signals_and_ohlc_df
