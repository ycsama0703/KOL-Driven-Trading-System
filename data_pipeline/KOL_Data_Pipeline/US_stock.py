# -*- coding: utf-8 -*-
"""
us_equity_name_lib.py

功能：
1. 从 Nasdaq 官方 symboldirectory 构建美股名称库：
   - https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt
   - https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt
2. 清洗公司名称，生成多个别名（alias），方便之后做匹配。
3. 保存为 parquet/csv 方便后续加载。

依赖：
    pip install pandas rapidfuzz pyarrow
"""

import re
import pandas as pd
from typing import List
from rapidfuzz import fuzz, process


# ======================
# 1. 下载 + 清洗基础列表
# ======================

def load_raw_symbol_tables() -> pd.DataFrame:
    """
    从 Nasdaq 官方下载 nasdaqlisted + otherlisted，并合并成一个 DataFrame。
    只做非常基础的清洗：去掉 Test Issue。
    """
    # 官方文件（“|” 分隔，最后一行是“File Creation Time”需要丢掉）
    url_nasdaq = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    url_other = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

    # 读入 NASDAQ
    nasdaq = pd.read_csv(url_nasdaq, sep="|")
    # 丢掉最后一行 summary（通常 Security Name 为 NaN 或 'File Creation Time:'）
    nasdaq = nasdaq[nasdaq["Symbol"].notna()]
    # 去掉 Test Issue
    if "Test Issue" in nasdaq.columns:
        nasdaq = nasdaq[nasdaq["Test Issue"] == "N"]

    nasdaq = nasdaq.rename(columns={
        "Symbol": "ticker",
        "Security Name": "security_name"
    })
    nasdaq["exchange"] = "NASDAQ"

    # 读入 Other-listed（NYSE, AMEX 等）
    other = pd.read_csv(url_other, sep="|")
    other = other[other["ACT Symbol"].notna()]
    if "Test Issue" in other.columns:
        other = other[other["Test Issue"] == "N"]

    other = other.rename(columns={
        "ACT Symbol": "ticker",
        "Security Name": "security_name"
    })
    other["exchange"] = "OTHER"  # NYSE/AMEX 等混在一起

    df = pd.concat([nasdaq[["ticker", "security_name", "exchange"]],
                    other[["ticker", "security_name", "exchange"]]],
                   ignore_index=True)

    # 去重
    df = df.drop_duplicates(subset=["ticker", "security_name"])

    # 只保留字符串类型
    df["ticker"] = df["ticker"].astype(str)
    df["security_name"] = df["security_name"].astype(str)

    return df


# ======================
# 2. 生成“干净公司名 + 别名”
# ======================

COMMON_SUFFIX_PATTERNS = [
    r" - Class [A-Z].*",
    r" - American Depositary Shares?.*",
    r" - American Depository Shares?.*",
    r" - Common Stock.*",
    r" - Ordinary Shares?.*",
    r" - ADR.*",
    r" - ADS.*",
    r" - Units?.*",
    r" - Warrant[s]?.*",
    r" - Rights?.*",
    r" - ETF.*",
]

LEGAL_SUFFIXES = [
    "Inc", "Inc.", "Corporation", "Corp", "Corp.", "Co", "Co.",
    "Ltd", "Ltd.", "Limited", "PLC", "S.A.", "N.V.", "PLC.", "plc"
]


def clean_base_name(security_name: str) -> str:
    """
    从 Security Name 中抽取一个“基础公司名”，用于构造别名。
    例：
        'Apple Inc. - Common Stock' -> 'Apple Inc.'
        'NVIDIA Corporation - Common Stock' -> 'NVIDIA Corporation'
    """
    name = security_name

    # 1) 去掉 “- Common Stock / - Ordinary Shares / - Class A …” 等后缀
    for pat in COMMON_SUFFIX_PATTERNS:
        name = re.sub(pat, "", name, flags=re.IGNORECASE)

    # 2) 去掉多余空格
    name = re.sub(r"\s+", " ", name).strip()

    return name


def generate_aliases_for_name(base_name: str) -> List[str]:
    """
    给一个基础公司名生成多个别名：
    - 原始 base_name
    - 去掉 , Inc. / Inc / Corporation / Corp 等
    - 去掉标点
    """
    aliases = set()

    # 原始
    aliases.add(base_name)

    # 1) 去掉逗号（如 'Apple Inc.' / 'Apple, Inc.'）
    no_comma = base_name.replace(",", " ")
    aliases.add(no_comma)

    # 2) 去掉公司后缀（Inc, Corp, Ltd, PLC 等）
    tokens = no_comma.split()
    # 从尾部开始删除合法后缀
    while tokens and tokens[-1].rstrip(".") in [s.rstrip(".") for s in LEGAL_SUFFIXES]:
        tokens.pop()
    stripped = " ".join(tokens)
    if stripped:
        aliases.add(stripped)

    # 3) 全部小写版本（方便做 case-insensitive 匹配）
    aliases.update({a.lower() for a in list(aliases)})

    return list(aliases)


def build_us_equity_name_lib() -> pd.DataFrame:
    """
    构建一个包含别名的名称库，每一行代表一个“ticker + 一个 alias”：
        columns: [ticker, exchange, security_name, base_name, alias]
    """
    base_df = load_raw_symbol_tables()

    base_df["base_name"] = base_df["security_name"].apply(clean_base_name)

    # 展开 alias
    rows = []
    for _, row in base_df.iterrows():
        ticker = row["ticker"]
        exch = row["exchange"]
        security_name = row["security_name"]
        base_name = row["base_name"]

        aliases = generate_aliases_for_name(base_name)
        for alias in aliases:
            rows.append({
                "ticker": ticker,
                "exchange": exch,
                "security_name": security_name,
                "base_name": base_name,
                "alias": alias
            })

    alias_df = pd.DataFrame(rows).drop_duplicates()

    return alias_df


def main_build_and_save(output_path: str = "us_equity_name_lib.csv"):
    alias_df = build_us_equity_name_lib()
    print("Total alias rows:", len(alias_df))
    # 保存为 parquet（更小更快），也可以改成 csv
    alias_df.to_csv(output_path, index=False)
    print("Saved to:", output_path)


if __name__ == "__main__":
    main_build_and_save()
