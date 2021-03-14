from typing import Dict

from backtesting.backtester import BackTester
from bollinger_bands_simple_breakout import BollingerBandsSimpleBreakout
from ma_double_cross_over import MADoubleCrossOver
from shared.BBHSSSignal import BBHSSSignal
from shared.Strategy import Strategy
from typing import Callable

if __name__ == "__main__":
    currencies = ['AAPL', 'MSFT', 'EURUSD=X', 'GBPUSD=X', 'AUDUSD=X',
                  'NZDUSD=X', 'GOOGL']

    strategies: Dict[Strategy, float] = {
        MADoubleCrossOver(
            magic_no=1,
            period_fast=5,
            period_slow=10
        ): 0.55,
        MADoubleCrossOver(
            magic_no=3,
            period_fast=50,
            period_slow=200
        ): 0.99,
        BollingerBandsSimpleBreakout(
            magic_no=2,
            period=14
        ): 0.0
    }

    risk: Dict[str, Callable[[float], bool]] = {
        'enter_buy': lambda current_signal: current_signal >= BBHSSSignal.buy.value,
        'enter_sell': lambda current_signal: current_signal <= BBHSSSignal.sell.value,
        'exit_buy': lambda current_signal: current_signal <= BBHSSSignal.hold.value,
        'exit_sell': lambda current_signal: current_signal >= BBHSSSignal.hold.value
    }

    for currency in currencies[:1]:
        print("Computing: " + currency)
        print(
            BackTester(
                currency,
                strategies=strategies,
                risk=risk
            ).run()
        )
