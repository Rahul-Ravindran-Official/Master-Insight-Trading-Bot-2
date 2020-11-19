import yfinance as yf
import datetime as dt


def obtain_ohlc_data(ticker: str):
    return yf.download(
        interval='1d',
        tickers=ticker,
        start=dt.date.today() - dt.timedelta(7000),
        end=dt.datetime.today()
    )

