# Scan all path's csv names


import os
import glob
import pandas as pd
import json
import random
import string



current_dir = os.path.dirname(__file__)
pattern = os.path.join("./data", "*.csv")
files = sorted(glob.glob(pattern))

# Read or create the shuffle_path dictionary
shuffle_path = "./dictionary.jsonl"
symbol_map = {}
if os.path.exists(shuffle_path):
    with open(shuffle_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if isinstance(entry, dict):
                    symbol_map.update(entry)
            except Exception:
                continue

def random_symbol(length=5):
    return ''.join(random.choices(string.ascii_uppercase, k=length))

# Ensure every basename has a symbol
updated = False
for fp in files:
    basename = os.path.basename(fp)
    if basename not in symbol_map:
        # Generate a unique random symbol
        while True:
            sym = random_symbol()
            if sym not in symbol_map.values():
                break
        symbol_map[basename] = sym
        updated = True

# Write back to shuffle_path if updated
if updated or not os.path.exists(shuffle_path):
    with open(shuffle_path, "w", encoding="utf-8") as f:
        for k, v in symbol_map.items():
            f.write(json.dumps({k: v}, ensure_ascii=False) + "\n")


output_file = os.path.join(current_dir, "AI-Trader/data/merged.jsonl")

with open(output_file, "w", encoding="utf-8") as fout:
    for fp in files:
        basename = os.path.basename(fp)
        data = pd.read_csv(fp)
        series = {}
        newSeries = {}
        for index, row in data.iterrows():
            day = {}
            day["1. buy price"] = round(row["Open"], 2)
            day["4. sell price"] = round(row["Close"], 2)
            day["3. low"] = round(row["Low"], 2)
            day["2. high"] = round(row["High"], 2)
            day["5. volume"] = row["Volume"]
            series[row["Date"]] = day 

            newDay = {}
            newDay["1. open"] = row["Open"]
            newDay["2. high"] = row["High"]
            newDay["3. low"] = row["Low"]
            newDay["4. close"] = row["Close"]
            newDay["5. volume"] = row["Volume"]
            newSeries[row["Date"]] = day 

        res = {}
        information = {}
        information["1. Information"] = "Daily Prices (buy price, high, low, sell price) and Volumes"
        information["2. Symbol"] = symbol_map[basename]
        res["Meta Data"] = information
        res["Time Series (Daily)"] = series
        fout.write(json.dumps(res, ensure_ascii=False) + "\n")

        newPath = "./AI-Trader/data/daily_prices_" + symbol_map[basename] + ".json"
        information["1. Information"] = "Daily Prices (open, high, low, close) and Volumes"
        res = {}
        res["Meta Data"] = information
        res["Time Series (Daily)"] = newSeries

        with open(newPath, "w", encoding="utf-8") as finOut:
            finOut.write(json.dumps(res, ensure_ascii=False) + "\n")

        



