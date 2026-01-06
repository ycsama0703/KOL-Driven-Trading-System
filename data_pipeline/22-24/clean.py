from pathlib import Path
from collections import defaultdict
import csv

# Increase max field size to handle very long text cells
csv.field_size_limit(10_000_000)

REQUIRED_FIELDS = ("excerpt", "confidence", "sentiment")
OUTPUT_FILE = "missing_output.csv"


def has_empty(row: dict) -> bool:
    """Return True if any cell in the row is empty or whitespace."""
    return any(v is None or str(v).strip() == "" for v in row.values())


def has_missing_required(row: dict) -> bool:
    """Return True if any of the required fields is empty or missing."""
    for field in REQUIRED_FIELDS:
        val = row.get(field)
        if val is None or str(val).strip() == "":
            return True
    return False


def aggregate_rows(rows, fields):
    """
    Collect non-empty values for each column across all rows of a video_id.
    Useful to see all fragments together when some rows are partially empty.
    """
    agg = {col: [] for col in fields}
    for row in rows:
        for col in fields:
            val = row.get(col)
            if val is not None:
                text = str(val).strip()
                if text:
                    agg[col].append(text)
    return {k: v for k, v in agg.items() if v}


def process_dir(folder: str = "22-24") -> None:
    """Scan all CSV files under folder and write missing info to a CSV file."""
    results = []
    for path in Path(folder).glob("*.csv"):
        rows_by_vid = defaultdict(list)
        missing = set()

        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue

            fields = reader.fieldnames
            for row in reader:
                vid = row.get("video_id") or row.get("videoId") or row.get("video") or "__missing_video_id__"
                rows_by_vid[vid].append(row)
                if has_missing_required(row):
                    missing.add(vid)

        if not missing:
            continue

        for vid in sorted(missing):
            rows = rows_by_vid[vid]
            title = next((r.get("title", "") for r in rows if r.get("title")), "")
            companies = []
            for row in rows:
                val = row.get("company")
                if val is not None:
                    text = str(val).strip()
                    if text:
                        companies.append(text)

            missing_fields = []
            for field in REQUIRED_FIELDS:
                if any(row.get(field) is None or str(row.get(field)).strip() == "" for row in rows):
                    missing_fields.append(field)

            results.append(
                {
                    "file": path.name,
                    "video_id": vid,
                    "title": title,
                    "rows": len(rows),
                    "missing_fields": ", ".join(missing_fields),
                    "company_combined": " | ".join(companies) if companies else "",
                }
            )

    if not results:
        print("No videos with missing required fields.")
        return

    with Path(OUTPUT_FILE).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "video_id", "title", "rows", "missing_fields", "company_combined"],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote {len(results)} record(s) to {OUTPUT_FILE}")


if __name__ == "__main__":
    process_dir()
