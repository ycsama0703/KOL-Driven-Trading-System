import pandas as pd


csv_path = "./agent/raw/positions_test_investwithhenry.csv"
df = pd.read_csv(csv_path)
tickers = set(df.iloc[:, 1])

# print(" ".join(tickers))
# tickers = {ticker for ticker in tickers if ticker != "BRK.B"}
print(tickers)

import yfinance as yf
dat = yf.download(
    tickers,
    period="1y",
    interval="1d",
    start="2024-01-01",
    group_by="ticker",
    auto_adjust=True,
)

for ticker in dat.columns.levels[0]:
    df_single = dat[ticker].copy()
    df_single.to_csv(f"./data/{ticker}.csv")
    print(f"Saved: {ticker}.csv")

