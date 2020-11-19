from backtesting.backtester import BackTester
from ma_double_cross_over import MADoubleCrossOver

currencies = ['AAPL', 'MSFT', 'EURUSD=X', 'GBPUSD=X', 'AUDUSD=X', 'NZDUSD=X', 'GOOGL']

for currency in currencies[:]:
    print("Computing: " + currency)
    print(
        BackTester(
            currency,
            {
                MADoubleCrossOver(5, 10): 0.5,
                MADoubleCrossOver(20, 50): 0.5
            }
        ).run()
    )
