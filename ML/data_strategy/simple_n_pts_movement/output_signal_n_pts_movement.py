import random

from market_data.ohlc_data import obtain_ohlc_data
import numpy as np
import pandas as pd


symbol = "EURUSD=X"

ohlc_data = obtain_ohlc_data(symbol, include_all_indicators=True)


p_32 = ohlc_data["Adj Close"][32:].to_numpy()
p_30 = ohlc_data["Adj Close"][30:].to_numpy()[:-2]
p_28 = ohlc_data["Adj Close"][28:].to_numpy()[:-4]
p_avg = (p_32 + p_30 + p_28) / 3

m = p_avg - ohlc_data["Adj Close"][:-32].to_numpy()

d = pd.DataFrame(m)

pd.DataFrame.describe(d)
print(pd.DataFrame.describe(d))

good_points = list(np.where(m > 0.0635)[0])
bad_points = np.intersect1d(list(np.where(m > 0)[0]), list(np.where(m < 0.05)[0]))
bad_points = random.sample(list(bad_points), 400)
worst_points = list(np.where(m < -0.0635)[0])

# Final Sample
sorted_sample = sorted(good_points+bad_points+worst_points)
final_df = ohlc_data.iloc[sorted_sample]

verdict = []
for e in sorted_sample:
    if e in good_points:
        verdict.append(1)
    else:
        verdict.append(0)

final_df = final_df.fillna(0)


pd.DataFrame(final_df).to_csv(
    r'/Users/rahul/Main/CloudStation/Spizen/spizen-forex/master-insight-trading-bot-2/ML/data_strategy/simple_n_pts_movement/A.csv',
    index=False,
    header=True
)

pd.DataFrame(verdict).to_csv(
    r'/Users/rahul/Main/CloudStation/Spizen/spizen-forex/master-insight-trading-bot-2/ML/data_strategy/simple_n_pts_movement/b.csv',
    index=False,
    header=True
)

pd.DataFrame(d.iloc[sorted_sample]).to_csv(
    r'/Users/rahul/Main/CloudStation/Spizen/spizen-forex/master-insight-trading-bot-2/ML/data_strategy/simple_n_pts_movement/value.csv',
    index=False,
    header=True
)
