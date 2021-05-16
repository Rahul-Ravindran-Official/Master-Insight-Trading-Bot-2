from market_data.ohlc_data import obtain_ohlc_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = obtain_ohlc_data('MSFT', include_all_indicators=True)[:]

b = dataset["Open"]

a = 9

for c in dataset.columns:
    if str(c).__contains__("CDL"):
        x = np.array(dataset["Close"])
        p = np.where(np.array(dataset[c]) == 100)[0]
        t = np.where(np.array(dataset[c]) == -100)[0]
        plt.plot(x)
        plt.plot(p, x[p], "v")
        plt.plot(t, x[t], "^")
        plt.show()
