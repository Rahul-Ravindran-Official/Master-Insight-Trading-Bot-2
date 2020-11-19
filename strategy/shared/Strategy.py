from typing import Dict, Tuple

import pandas as pd


class Strategy:
    def get_bbhss_signal(self, ohlc_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        pass

    def process_to_bbhss(self, signals_and_ohlc_df: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        pass
