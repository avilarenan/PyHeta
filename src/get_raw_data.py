import yfinance as yf
from typing import Dict
import pandas as pd
import json
import btgsolutions_dataservices as btg

tickers = ["PETR4.SA", "VALE3.SA", "ITUB4.SA"]

start_date = "2022-01-01"
end_date = "2023-06-22"

def download_financial_raw_data_batch(tickers: Dict[str, str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """
    Example call:

    ```
    download_financial_raw_data({
            "PETR4_raw_data": "PETR4.SA",
            "VALE3_raw_data": "VALE3.SA"
        },
        start_date="2017-01-01",
        end_date="2023-07-01"
    )
    ```

    Returns Dict mapping tickers key to DataFrame
    """
    ret = {}
    for filename, ticker in tickers.items():
        df = download_financial_raw_data_single(ticker, start_date, end_date, filename)
        ret[filename] = df

    return ret

def download_financial_raw_data_single(ticker: str, start_date: str, end_date: str, filename: str, base_path: str) -> pd.DataFrame:
    """
    Example call: download_financial_raw_data("PETR4.SA", start_date="2017-01-01", end_date="2023-07-01")

    Filename must not include csv extension.
    """
    data = yf.download(f"{ticker}", start=start_date, end=end_date, prepost=False, repair=True, auto_adjust=True)
    data.to_csv(f"{base_path}/{filename}.csv")
    return data

def download_financial_raw_data_single_solutions_dataservices(ticker: str, start_date: str, end_date: str, filename: str, base_path: str, credentials_path: str = None):
    try:
        credential_name = 'SOLUTIONS_DATASERVICES_API_KEY'
        f = open(credentials_path)
        data = json.load(f)
        API_KEY = data[credential_name]
    except Exception as e:
        raise(f"Error while trying to find credentials for {credential_name}: \n{e} \nPlease, create a credentials.json file at {credentials_path} with the value for SOLUTIONS_DATASERVICES_API_KEY.")

    if ticker.endswith(".SA"):
        ticker = ticker[:-3]

    hist_candles = btg.HistoricalCandles(api_key=API_KEY)
    df = hist_candles.get_interday_history_candles(ticker=ticker,  market_type='stocks', corporate_events_adj=True, start_date=start_date, end_date=end_date, rmv_after_market=True, timezone='UTC', raw_data=False)

    df = df[["open_price", "high_price", "low_price", "close_price", "volume", "date"]].copy()
    df = df.rename(
        columns={
            "open_price" : "Open",
            "high_price" : "High",
            "low_price" : "Low",
            "close_price" : "Close",
            "volume" : "Volume",
            "date" : "Date"
        }
    )
    
    df["Adj Close"] = df["Close"]

    df = df.set_index("Date")
    df.to_csv(f"{base_path}/{filename}.csv")
    return df