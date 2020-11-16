import yfinance as yf
import datetime as dt


def obtain_ohlc_data(ticker: str):
    return yf.download(
        tickers=ticker,
        start=dt.date.today() - dt.timedelta(1825),
        end=dt.datetime.today()
    )

