from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd


def _load_first_video_metadata(base_dir: Path) -> Dict[str, Any]:
    """Read the first row from the channel Excel to get id/title/description."""
    excel_files = sorted((base_dir / "youtube_kol_outputs").glob("*.xlsx"))
    if not excel_files:
        raise FileNotFoundError("No Excel files found in youtube_kol_outputs.")

    df = pd.read_excel(excel_files[0])
    if df.empty:
        raise ValueError(f"{excel_files[0].name} is empty.")

    row = df.iloc[0]
    return {
        "video_id": str(row.get("video_id", "")).strip(),
        "title": str(row.get("title", "")).strip(),
        "description": str(row.get("description", "")).strip(),
        "channel_name": str(row.get("channel_name", "")).strip(),
    }


def _load_summary_for_video(base_dir: Path, video_id: str) -> str:
    """
    Attempt to pull summary text for the video from the big subtitles CSV.
    Returns empty string if not found.
    """
    summary_csv = base_dir.parent / "input" / "youtube_subtitles_Ales_World_of_Stocks(By Gemini).csv"
    if not summary_csv.exists():
        return ""

    df = pd.read_csv(summary_csv, dtype=str, usecols=["video_id", "summary"])
    match = df[df["video_id"] == video_id]
    if match.empty:
        return ""
    return str(match.iloc[0]["summary"])


def _load_company_mentions(base_dir: Path, channel_name: str, video_id: str) -> List[Dict[str, Any]]:
    """Load company mention JSON for the given video."""
    json_path = base_dir / "youtube_kol_outputs" / f"{channel_name}_{video_id}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Company JSON not found: {json_path.name}")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {json_path.name}, got {type(data)}")
    return data


def combine_first_video() -> Path:
    """
    Combine the first video's metadata with its company mentions into a CSV.
    Output columns: video_id, title, description, summary, company, excerpt, confidence, sentiment.
    """
    base_dir = Path(__file__).resolve().parent

    meta = _load_first_video_metadata(base_dir)
    video_id = meta["video_id"]
    channel_name = meta["channel_name"]

    summary_text = _load_summary_for_video(base_dir, video_id)
    companies = _load_company_mentions(base_dir, channel_name, video_id)

    rows = []
    for item in companies:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "video_id": video_id,
                "title": meta["title"],
                "description": meta["description"],
                "summary": summary_text,
                "company": item.get("company", ""),
                "excerpt": item.get("excerpt", ""),
                "confidence": item.get("confidence", ""),
                "sentiment": item.get("sentiment", ""),
            }
        )

    if not rows:
        raise ValueError("No company mentions found to combine.")

    out_path = base_dir / "youtube_kol_outputs" / "first_video_combined.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    out = combine_first_video()
    print(f"Wrote combined data to: {out}")

