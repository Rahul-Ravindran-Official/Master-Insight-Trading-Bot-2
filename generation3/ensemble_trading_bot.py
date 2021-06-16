from typing import Dict, List
import numpy as np


class SignalProvider:

    def get_signal(self) -> np.array:
        pass

    def get_buy_signal(self) -> np.array:
        pass

    def get_sell_signal(self) -> np.array:
        pass


class EnsembleTradingBot:
    signal_providers: Dict[SignalProvider, float] # Signal, weightage
    master_signal: np.array

    def __init__(self, signal_providers: Dict[SignalProvider, float]):
        self.signal_providers = signal_providers

    def master_signal(self):
        for sp in self.signal_providers.keys():
            self.master_signal += np.multiply(sp.get_signal(),self.signal_providers[sp])
