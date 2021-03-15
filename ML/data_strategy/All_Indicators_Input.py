import talib

def all_technical_indicator_inputs(ohlc_data):
    ohlc_data["SMA-10"] = talib.SMA(ohlc_data["Adj Close"], 10)
    ohlc_data["SMA-21"] = talib.SMA(ohlc_data["Adj Close"], 21)
    ohlc_data["SMA-50"] = talib.SMA(ohlc_data["Adj Close"], 50)
    ohlc_data["SMA-100"] = talib.SMA(ohlc_data["Adj Close"], 100)
    ohlc_data["SMA-200"] = talib.SMA(ohlc_data["Adj Close"], 200)
    ohlc_data["SMA-250"] = talib.SMA(ohlc_data["Adj Close"], 250)
    bb = talib.BBANDS(ohlc_data["Adj Close"])
    ohlc_data["BBANDS--0"] = bb[0]
    ohlc_data["BBANDS--1"] = bb[1]
    ohlc_data["BBANDS--2"] = bb[2]
    ohlc_data["RSI"] = talib.RSI(ohlc_data["Adj Close"], 14)
    ohlc_data["CCI-20"] = talib.CCI(ohlc_data["High"], ohlc_data["Low"], ohlc_data["Close"], 20)
    ohlc_data["CCI"] = talib.CCI(ohlc_data["High"], ohlc_data["Low"], ohlc_data["Close"])

    stock = talib.STOCH(ohlc_data["High"], ohlc_data["Low"], ohlc_data["Close"])
    ohlc_data["STOCH--0"] = stock[0]
    ohlc_data["STOCH--1"] = stock[1]

    ohlc_data["AD"] = talib.AD(ohlc_data["High"], ohlc_data["Low"], ohlc_data["Close"], ohlc_data["Volume"])

    macd = talib.MACD(ohlc_data["Adj Close"], 12, 26, 9)
    ohlc_data["MACD--0"] =macd[0]
    ohlc_data["MACD--1"] =macd[1]
    ohlc_data["MACD--2"] =macd[2]

    ohlc_data["EMA-10"] = talib.EMA(ohlc_data["Adj Close"], 10)
    ohlc_data["EMA-21"] = talib.EMA(ohlc_data["Adj Close"], 21)
    ohlc_data["EMA-50"] = talib.EMA(ohlc_data["Adj Close"], 50)
    ohlc_data["EMA-100"] = talib.EMA(ohlc_data["Adj Close"], 100)
    ohlc_data["EMA-200"] = talib.EMA(ohlc_data["Adj Close"], 200)
    ohlc_data["EMA-250"] = talib.EMA(ohlc_data["Adj Close"], 250)
    ohlc_data["ATR"] = talib.ATR(ohlc_data["High"], ohlc_data["Low"], ohlc_data["Close"])
    ohlc_data["MFI"] = talib.MFI(ohlc_data["High"], ohlc_data["Low"], ohlc_data["Close"], ohlc_data["Volume"])
    ohlc_data["ADX"] = talib.ADX(ohlc_data["High"], ohlc_data["Low"], ohlc_data["Close"])
    ohlc_data["SAR"] = talib.SAR(ohlc_data["High"], ohlc_data["Low"])
    ohlc_data["OBV"] = talib.OBV(ohlc_data["Adj Close"], ohlc_data["Volume"])
    ohlc_data["PPO"] = talib.PPO(ohlc_data["Adj Close"])
    ohlc_data["DX"] = talib.DX(ohlc_data["High"], ohlc_data["Low"], ohlc_data["Close"])

    ohlc_data.bfill()

    return ohlc_data

if __name__ == "__main__":
    all_technical_indicator_inputs()
