from __future__ import annotations

import csv
import json
import ast
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

# Fields that must be present for a row to be considered complete
REQUIRED_FIELDS = ("excerpt", "confidence", "sentiment")

# Allow very large text cells
csv.field_size_limit(10_000_000)


def has_missing_required(row: Dict[str, Any]) -> bool:
    """Return True if any required field is missing or blank in the row."""
    for field in REQUIRED_FIELDS:
        val = row.get(field)
        if val is None or str(val).strip() == "":
            return True
    return False


def parse_company_payload(text: str) -> List[Dict[str, Any]]:
    """
    Parse the combined company text into a list of dicts.
    Handles simple JSON or Python-literal style payloads.
    """
    if not text:
        return []

    cleaned = text.replace("```json", "").replace("```", "").strip()
    if not cleaned or cleaned == "[]":
        return []

    data = None
    try:
        data = json.loads(cleaned)
    except Exception:
        try:
            data = ast.literal_eval(cleaned)
        except Exception:
            return []

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []

    return [item for item in data if isinstance(item, dict)]


def first_non_empty(rows: List[Dict[str, Any]], field: str) -> str:
    """Return the first non-empty value for a field across rows."""
    for row in rows:
        val = row.get(field)
        if val is not None:
            text = str(val).strip()
            if text:
                return text
    return ""


def process_file(path: Path) -> None:
    """Process a single CSV file and write cleaned output."""
    output_rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print(f"{path.name}: missing header or empty file; skipped")
            return
        fieldnames = reader.fieldnames
        rows_by_vid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for row in reader:
            vid = row.get("video_id") or row.get("videoId") or row.get("video") or "__missing_video_id__"
            rows_by_vid[vid].append(row)

    for vid, rows in rows_by_vid.items():
        has_missing = any(has_missing_required(r) for r in rows)

        if not has_missing:
            # Keep original rows as-is
            output_rows.extend(rows)
            continue

        # Combine company fragments
        company_parts = []
        for r in rows:
            val = r.get("company")
            if val is not None:
                text = str(val).strip()
                if text:
                    company_parts.append(text)

        combined_company = "\n".join(company_parts).strip()
        company_records = parse_company_payload(combined_company)
        if not company_records:
            # No usable company data; skip this video
            continue

        # Base info pulled from first non-empty occurrence per field
        base_info = {}
        for field in fieldnames:
            if field in ("company", "excerpt", "confidence", "sentiment"):
                continue
            base_info[field] = first_non_empty(rows, field)

        for record in company_records:
            new_row = {col: "" for col in fieldnames}
            new_row.update(base_info)
            new_row["company"] = record.get("company", "")
            new_row["excerpt"] = record.get("excerpt", "")
            new_row["confidence"] = record.get("confidence", "")
            new_row["sentiment"] = record.get("sentiment", "")
            output_rows.append(new_row)

    if not output_rows:
        print(f"{path.name}: no data to write.")
        return

    out_path = path.with_name(f"{path.stem}_cleaned.csv")
    with out_path.open("w", encoding="utf-8-sig", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote cleaned file: {out_path.name} (rows: {len(output_rows)})")


def main():
    for path in Path("22-24").glob("*.csv"):
        process_file(path)


if __name__ == "__main__":
    main()
