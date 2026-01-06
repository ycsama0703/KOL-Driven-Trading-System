def find_company(content):
    from transformers import pipeline

    ner = pipeline("ner", grouped_entities=True, model="dslim/bert-base-NER")

    def extract_orgs(text):
        entities = ner(text)
        return [e["word"] for e in entities if e["entity_group"] == "ORG"]

    # text = "Goldman Sachs thinks Apple and Amazon will benefit from AI."
    # print(extract_orgs(text))

    import pandas as pd
    from typing import List, Dict, Any
    from rapidfuzz import process, fuzz

    def load_name_lib(path) -> pd.DataFrame:
        df = pd.read_csv(path)
        # 只保留我们需要的列
        df = df[["ticker", "exchange", "base_name", "alias"]].drop_duplicates()
        return df

    def match_companies_in_text(
            text: str,
            name_lib: pd.DataFrame,
            top_k: int = 5,
            score_threshold: int = 85
    ) -> List[Dict[str, Any]]:
        """
        在 text 中查找可能出现的公司：
        - 使用 alias 字段做模糊匹配（partial_ratio）
        - 返回去重后的 ticker 列表及匹配信息

        返回格式：[
            {
                "ticker": "AAPL",
                "base_name": "Apple Inc.",
                "best_alias": "apple inc",
                "score": 96.5,
                "exchange": "NASDAQ"
            },
            ...
        ]
        """
        # 为了速度，先准备 alias -> 行索引 的 mapping
        alias_list = name_lib["alias"].tolist()

        # 直接对整段 text 做匹配（也可以先做 NER 抽公司名再对每个短 phrase 做匹配）
        matches = process.extract(
            text,
            alias_list,
            scorer=fuzz.partial_ratio,
            limit=top_k * 10  # 多取一点，后面再根据 score 和去重过滤
        )

        results = []
        seen_tickers = set()

        for alias_str, score, alias_idx in matches:
            if score < score_threshold:
                continue

            row = name_lib.iloc[alias_idx]
            ticker = row["ticker"]

            if ticker in seen_tickers:
                continue
            seen_tickers.add(ticker)

            results.append({
                "ticker": ticker,
                "base_name": row["base_name"],
                "best_alias": row["alias"],
                "score": float(score),
                "exchange": row["exchange"],
            })

            if len(results) >= top_k:
                break

        return results

    lib = load_name_lib("us_equity_name_lib.csv")
    matches = match_companies_in_text(content, lib, top_k=10, score_threshold=80)
    return matches
