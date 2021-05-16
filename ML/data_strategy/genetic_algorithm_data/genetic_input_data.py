import talib
from market_data.ohlc_data import obtain_ohlc_data
import pandas as pd
from scipy.signal import lfilter

def smooth_algo(data, cnt: int = 200, smooth_intensity: int = 3):
    data = data[:cnt]
    n = smooth_intensity  # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
    yy = lfilter(b, a, data)[n - 1:]
    return yy

symbol = "EURUSD=X"

ohlc_data = obtain_ohlc_data(symbol, include_all_indicators=True)
smoothened_data = smooth_algo(ohlc_data["Adj Close"], 4500, 2)

p_3 = smoothened_data[3:]
p_7 = smoothened_data[7:]
p_21 = smoothened_data[21:]
p_50 = smoothened_data[50:]

diff_p_3 = (p_3 - smoothened_data[:-3])[:-47]
diff_p_7 = (p_7 - smoothened_data[:-7])[:-43]
diff_p_21 = (p_21 - smoothened_data[:-21])[:-29]
diff_p_50 = (p_50 - smoothened_data[:-50])

df = pd.DataFrame()
df['P3'] = diff_p_3
df['P7'] = diff_p_7
df['P21'] = diff_p_21
df['P50'] = diff_p_50

print(pd.DataFrame.describe(df))

# Input Signals
# RSI
# Price - EMA 3, 7, 21, 50

signals = pd.DataFrame()
signals["EMA-3"] = smoothened_data - talib.EMA(smoothened_data, 3)
signals["EMA-7"] = smoothened_data - talib.EMA(smoothened_data, 7)
signals["EMA-21"] = smoothened_data - talib.EMA(smoothened_data, 21)
signals["EMA-50"] = smoothened_data - talib.EMA(smoothened_data, 50)
signals["RSI"] = talib.RSI(smoothened_data, 14)
signals["RSI"] = signals["RSI"]/100

signals = signals[50:-50]
df = df[50:]


# Exporting

pd.DataFrame(df).to_csv(
    r'/Users/rahul/Main/CloudStation/Spizen/spizen-forex/master-insight-trading-bot-2/ML/data_strategy/genetic_algorithm_data/output.csv',
    index=False,
    header=True
)

pd.DataFrame(signals).to_csv(
    r'/Users/rahul/Main/CloudStation/Spizen/spizen-forex/master-insight-trading-bot-2/ML/data_strategy/genetic_algorithm_data/signals.csv',
    index=False,
    header=True
)

