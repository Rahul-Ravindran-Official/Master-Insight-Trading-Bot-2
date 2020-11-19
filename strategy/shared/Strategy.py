from typing import Dict

import pandas as pd


class Strategy:
    def get_signals(self, ohlc_df: pd.DataFrame) -> pd.DataFrame:
        pass

    def process_to_bbhss(self, signals_and_ohlc_df: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        pass
