import json
import pandas as pd
import os


current_dir = os.path.dirname(__file__)
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

output_file = os.path.join(current_dir, "symbols.csv")
with open(output_file, "w", encoding="utf-8") as fout:
    fout.write("\n".join(symbol_map.values()))

     

