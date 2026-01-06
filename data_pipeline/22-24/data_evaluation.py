from __future__ import annotations

import csv
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Columns that should always be present and non-empty in cleaned data
REQUIRED_COLS = ("video_id", "title", "company", "excerpt", "confidence", "sentiment")

# Expect confidence and sentiment to be numeric; set expected ranges
NUMERIC_COLS = {
    "confidence": (0.0, 1.0),
    "sentiment": (-1.0, 1.0),
}

# Increase limit to safely read long text fields
csv.field_size_limit(10_000_000)


def is_blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def to_float(value: Any) -> Tuple[bool, float | None]:
    try:
        return True, float(value)
    except Exception:
        return False, None


def evaluate_file(path: Path) -> Dict[str, Any]:
    """Evaluate a single cleaned CSV file and return metrics."""
    metrics: Dict[str, Any] = {
        "file": path.name,
        "total_rows": 0,
        "videos": 0,
        "missing_required_cells": 0,
        "invalid_numeric": 0,
        "out_of_range_numeric": 0,
        "empty_company": 0,
        "empty_excerpt": 0,
        "empty_title": 0,
        "empty_video_id": 0,
        "duplicate_rows": 0,
    }

    rows_by_vid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            metrics["error"] = "missing header"
            return metrics

        missing_cols = [col for col in REQUIRED_COLS if col not in reader.fieldnames]
        if missing_cols:
            metrics["error"] = f"missing required columns: {', '.join(missing_cols)}"
            return metrics

        for row in reader:
            metrics["total_rows"] += 1
            vid = row.get("video_id") or ""
            rows_by_vid[vid].append(row)

            # Required field checks
            for col in REQUIRED_COLS:
                if is_blank(row.get(col)):
                    metrics["missing_required_cells"] += 1
                    if col == "company":
                        metrics["empty_company"] += 1
                    elif col == "excerpt":
                        metrics["empty_excerpt"] += 1
                    elif col == "title":
                        metrics["empty_title"] += 1
                    elif col == "video_id":
                        metrics["empty_video_id"] += 1

            # Numeric validation
            for col, (low, high) in NUMERIC_COLS.items():
                val = row.get(col)
                if is_blank(val):
                    continue
                ok, number = to_float(val)
                if not ok:
                    metrics["invalid_numeric"] += 1
                    continue
                if number is not None and (number < low or number > high):
                    metrics["out_of_range_numeric"] += 1

    metrics["videos"] = len(rows_by_vid)

    # Duplicate detection (exact row duplicates)
    counter = Counter()
    for vid_rows in rows_by_vid.values():
        for r in vid_rows:
            key = tuple(r.get(col, "") for col in reader.fieldnames or [])
            counter[key] += 1
    metrics["duplicate_rows"] = sum(count - 1 for count in counter.values() if count > 1)

    return metrics


def write_report(results: List[Dict[str, Any]], out_path: Path) -> None:
    if not results:
        print("No cleaned CSV files found. Nothing to evaluate.")
        return

    fieldnames = [
        "file",
        "total_rows",
        "videos",
        "missing_required_cells",
        "invalid_numeric",
        "out_of_range_numeric",
        "empty_company",
        "empty_excerpt",
        "empty_title",
        "empty_video_id",
        "duplicate_rows",
        "error",
    ]

    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            # Ensure all fields exist for consistent CSV output
            payload = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(payload)

    print(f"Wrote evaluation report: {out_path.name} (files: {len(results)})")


def evaluate(folder: str = "22-24") -> None:
    """Evaluate all *_cleaned.csv files under the folder."""
    results: List[Dict[str, Any]] = []
    for path in Path(folder).glob("*_cleaned.csv"):
        metrics = evaluate_file(path)
        results.append(metrics)

    out_path = Path(folder) / "evaluation_report.csv"
    write_report(results, out_path)

    # Print quick summary
    for r in results:
        if r.get("error"):
            print(f"{r['file']}: ERROR -> {r['error']}")
        else:
            print(
                f"{r['file']}: rows={r['total_rows']} videos={r['videos']} "
                f"missing_cells={r['missing_required_cells']} invalid_numeric={r['invalid_numeric']} "
                f"out_of_range={r['out_of_range_numeric']} duplicates={r['duplicate_rows']}"
            )


if __name__ == "__main__":
    evaluate()
