from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple


def _iter_confidences(path: Path) -> Tuple[List[float], Counter]:
    """
    Collect all confidence values from JSON files under the given path.
    Returns the values and a per-file count of extracted confidences.
    """
    confidences: List[float] = []
    per_file_counts: Counter = Counter()

    for file_path in sorted(path.glob("*.json")):
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[WARN] Skip {file_path.name}: {exc}")
            continue

        if not isinstance(data, list):
            print(f"[WARN] Skip {file_path.name}: expected list root")
            continue

        values = []
        for item in data:
            if isinstance(item, dict) and "confidence" in item:
                val = item["confidence"]
                if isinstance(val, (int, float)):
                    values.append(float(val))

        confidences.extend(values)
        per_file_counts[file_path.name] = len(values)

    return confidences, per_file_counts


def _percentile(sorted_values: List[float], p: float) -> float:
    """Inclusive linear interpolation percentile in [0, 1]."""
    if not sorted_values:
        raise ValueError("Cannot compute percentile on empty list")
    if p < 0 or p > 1:
        raise ValueError("Percentile must be in [0, 1]")

    k = (len(sorted_values) - 1) * p
    lower = math.floor(k)
    upper = math.ceil(k)

    if lower == upper:
        return sorted_values[int(k)]

    low_val = sorted_values[lower]
    high_val = sorted_values[upper]
    return low_val + (high_val - low_val) * (k - lower)


def describe(values: Iterable[float]) -> dict:
    vals = list(values)
    if not vals:
        return {}

    sorted_vals = sorted(vals)
    return {
        "count": len(vals),
        "mean": statistics.fmean(vals),
        "median": statistics.median(sorted_vals),
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
        "stdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "p10": _percentile(sorted_vals, 0.10),
        "p25": _percentile(sorted_vals, 0.25),
        "p75": _percentile(sorted_vals, 0.75),
        "p90": _percentile(sorted_vals, 0.90),
    }


def histogram(values: Iterable[float], bins: int = 10) -> List[Tuple[str, int]]:
    """
    Produce a simple histogram bucketed between 0 and 1 inclusive.
    Returns a list of (bucket_label, count).
    """
    vals = list(values)
    if not vals:
        return []

    bin_counts = [0 for _ in range(bins)]
    for v in vals:
        idx = min(int(v * bins), bins - 1)
        bin_counts[idx] += 1

    result = []
    for i, count in enumerate(bin_counts):
        left = i / bins
        right = (i + 1) / bins
        label = f"[{left:.1f}, {right:.1f})" if i < bins - 1 else f"[{left:.1f}, 1.0]"
        result.append((label, count))
    return result


def main() -> None:
    data_dir = Path(__file__).resolve().parent / "youtube_kol_outputs"
    confidences, per_file_counts = _iter_confidences(data_dir)

    if not confidences:
        print("No confidence values found.")
        return

    stats = describe(confidences)

    print("=== Confidence Summary ===")
    print(f"Files scanned: {len(per_file_counts)}")
    print(f"Total confidences: {stats['count']}")
    print(f"Mean: {stats['mean']:.4f}")
    print(f"Median: {stats['median']:.4f}")
    print(f"Min: {stats['min']:.4f}")
    print(f"Max: {stats['max']:.4f}")
    print(f"Std dev (population): {stats['stdev']:.4f}")
    print(f"P10: {stats['p10']:.4f}, P25: {stats['p25']:.4f}, "
          f"P75: {stats['p75']:.4f}, P90: {stats['p90']:.4f}")

    print("\n=== Histogram (10 bins) ===")
    for label, count in histogram(confidences):
        print(f"{label}: {count}")

    top_missing = [name for name, cnt in per_file_counts.items() if cnt == 0]
    if top_missing:
        print("\nFiles without confidence entries:")
        for name in top_missing:
            print(f"- {name}")


if __name__ == "__main__":
    main()
