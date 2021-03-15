import yfinance as yf
import datetime as dt

from ML.data_strategy.All_Indicators_Input import all_technical_indicator_inputs


def obtain_ohlc_data(
        ticker: str,
        start: dt.datetime=dt.date.today() - dt.timedelta(1825),
        end: dt.datetime=dt.datetime.today(),
        include_all_indicators=False
):
    data = yf.download(
        interval='1d',
        tickers=ticker,
        start=start,
        end=end
    )

    if include_all_indicators:
        data = all_technical_indicator_inputs(data)

    return data

