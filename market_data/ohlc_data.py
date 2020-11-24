import yfinance as yf
import datetime as dt


def obtain_ohlc_data(
        ticker: str,
        start: dt.datetime=dt.date.today() - dt.timedelta(1825),
        end: dt.datetime=dt.datetime.today()
):
    return yf.download(
        interval='1d',
        tickers=ticker,
        start=start,
        end=end
    )

