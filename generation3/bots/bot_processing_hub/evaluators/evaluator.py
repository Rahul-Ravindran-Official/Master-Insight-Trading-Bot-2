import pandas as pd
from sklearn.preprocessing import minmax_scale
import math

ticker = "NZDUSD=X"

# Getting Data
adj_close = pd.read_csv('/Users/rahul/Main/CloudStation/Spizen/spizen-forex/master-insight-trading-bot-2/generation3/bots/datasets/' + ticker + '_clf_input_signals.csv')["Adj Close"]
buy_points = pd.read_csv('/Users/rahul/Main/CloudStation/Spizen/spizen-forex/master-insight-trading-bot-2/generation3/bots/datasets/' + ticker + '_clf_buy_points.csv')
buy_signals = pd.read_csv('/Users/rahul/Main/CloudStation/Spizen/spizen-forex/master-insight-trading-bot-2/generation3/bots/datasets/' + ticker + '_clf_output_signals.csv')
sell_points = pd.read_csv('/Users/rahul/Main/CloudStation/Spizen/spizen-forex/master-insight-trading-bot-2/generation3/bots/datasets/' + ticker + '_clf_sell_points.csv')

# Series Conversion
bp_series = list(buy_points.to_numpy().T[0])
sp_series = list(sell_points.to_numpy().T[0])

# Creating segments
all_points = sorted(set([0] + bp_series + sp_series + [len(adj_close)-1]))
buy_pos = 1
all_segments = []
for i in range(len(all_points)-2):
    idx_1 = all_points[i]
    idx_2 = all_points[i+1]+1
    if buy_pos == 1:
        all_segments.append(adj_close[idx_1:idx_2])
        buy_pos = 0
    else:
        all_segments.append(adj_close[idx_1:idx_2])
        buy_pos = 1

# Normalizing all buy sell segments
buy_pos = 1
normalised_segments = []

for segment in all_segments:
    normalised_segments.append(
        minmax_scale(
            list(segment),
            feature_range=(0, 1),
            axis=0,
            copy=True
        )
    )

# Making the Incorrect Prediction Array
incorrect_prediction_penalty = []
while len(normalised_segments) > 1:
    incorrect_prediction_penalty.extend(list(normalised_segments.pop(0)[:-1]))
incorrect_prediction_penalty.extend(list(normalised_segments.pop(0)[:]))


# Filler
filler_scores_required = len(buy_signals) - len(incorrect_prediction_penalty) + 1

normalised_fillers = minmax_scale(
    list(adj_close[-filler_scores_required:]),
    feature_range=(0, 1),
    axis=0,
    copy=True
)
incorrect_prediction_penalty.extend(normalised_fillers[1:])

# Sanity Check
assert len(buy_signals) == len(incorrect_prediction_penalty)

# Exporting penalty list
pd.DataFrame(incorrect_prediction_penalty).to_csv(
    r'./' + ticker + '_clf_penalty_buy_signals.csv',
    index=False,
    header=True
)

# Exporting proxy index list
pd.DataFrame(list(range(len(incorrect_prediction_penalty)))).to_csv(
    r'./' + ticker + '_clf_proxy_index.csv',
    index=False,
    header=True
)
