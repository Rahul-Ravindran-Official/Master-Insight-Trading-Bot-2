from typing import List, Tuple

import pandas as pd


class Indicator:
    def get_signal(self, ohlc_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        pass

