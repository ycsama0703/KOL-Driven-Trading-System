import os
import re
import time
import json
import pandas as pd
import subprocess
import google.generativeai as genai
from googleapiclient.discovery import build

# --- API Keys & Config ---
API_KEY_YT = os.getenv("YOUTUBE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"

# 2022-01-01 ~ 2024-12-31
PUBLISHED_AFTER = "2022-01-01T00:00:00Z"
PUBLISHED_BEFORE = "2022-03-31T23:59:59Z"

output_dir = "./output_test"
os.makedirs(output_dir, exist_ok=True)

# Update this list to control which KOLs are fetched/processed.
kol_list = [
    "MarketBeat",
    "Invest with Henry",
    "Everything Money",
]

_gemini_model = None


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")


def get_youtube_client():
    return build("youtube", "v3", developerKey=API_KEY_YT)


def get_channel_id(youtube_client, username: str):
    resp = youtube_client.search().list(
        q=username, type="channel", part="snippet", maxResults=1
    ).execute()
    return resp["items"][0]["snippet"]["channelId"] if resp.get("items") else None


def fetch_video_ids(youtube_client, channel_id: str):
    video_ids, token = [], None
    while True:
        resp = youtube_client.search().list(
            part="id",
            channelId=channel_id,
            publishedAfter=PUBLISHED_AFTER,
            publishedBefore=PUBLISHED_BEFORE,
            maxResults=50,
            order="date",
            type="video",
            pageToken=token,
        ).execute()
        video_ids += [
            item["id"].get("videoId", "")
            for item in resp.get("items", [])
            if "videoId" in item.get("id", {})
        ]
        token = resp.get("nextPageToken")
        if not token:
            break
        time.sleep(0.4)
    return [vid for vid in video_ids if vid]


def fetch_video_details(youtube_client, video_ids, channel_name: str):
    details = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i : i + 50]
        resp = youtube_client.videos().list(
            part="snippet,statistics", id=",".join(batch)
        ).execute()
        for item in resp.get("items", []):
            snip = item.get("snippet", {})
            stats = item.get("statistics", {})
            details.append(
                {
                    "channel_name": channel_name,
                    "video_id": item.get("id", ""),
                    "publishedAt": snip.get("publishedAt", ""),
                    "title": snip.get("title", ""),
                    "description": snip.get("description", ""),
                    "viewCount": stats.get("viewCount", ""),
                    "likeCount": stats.get("likeCount", ""),
                }
            )
        time.sleep(0.4)
    return details


def download_transcript(video_id):
    vtt_path = os.path.join(output_dir, f"{video_id}.en.vtt")
    try:
        cmd = [
            "yt-dlp",
            "--write-auto-sub",
            "--sub-lang",
            "en",
            "--skip-download",
            "-o",
            os.path.join(output_dir, "%(id)s.%(ext)s"),
            f"https://www.youtube.com/watch?v={video_id}",
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=False)

        if not os.path.exists(vtt_path):
            vtts = [
                f
                for f in os.listdir(output_dir)
                if f.startswith(video_id) and f.endswith(".vtt")
            ]
            if vtts:
                vtt_path = os.path.join(output_dir, vtts[0])
            else:
                return ""

        text_lines = []
        with open(vtt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("WEBVTT") or re.match(
                    r"^\d\d:\d\d:\d\d\.\d\d\d -->", line
                ):
                    continue
                cleaned_line = re.sub(r"<[^>]+>", "", line)
                cleaned_line = re.sub(r"http\S+", "", cleaned_line)
                text_lines.append(cleaned_line)

        deduped = []
        prev = ""
        for line in text_lines:
            if line != prev:
                deduped.append(line)
            prev = line

        return " ".join(deduped)
    finally:
        for f in os.listdir(output_dir):
            if f.startswith(video_id) and f.endswith(".vtt"):
                try:
                    os.remove(os.path.join(output_dir, f))
                except Exception:
                    pass


def find_company_llm(content, title: str = "", description: str = ""):
    """
    Use Gemini to extract distinct company names with a short supporting excerpt.
    Considers title/transcript/description (ignore ads/promos). Returns list of dicts.
    """
    global _gemini_model
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    if _gemini_model is None:
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_model = genai.GenerativeModel(GEMINI_MODEL)

    prompt = f"""
You are extracting company mentions.
- Use the title, description, and transcript to find distinct company names.
- Ignore ads/promo/CTA content when picking excerpts.
- For each company, return all the related sentences combined together from the transcript that references it.
- After extracting excerpt, score:
  * confidence: 0-1 (certainty the excerpt refers to that company)
  * sentiment: -1 to 1 (negative to positive tone toward the company)
- Output JSON array only, schema:
[{{"company": "<name>", "excerpt": "<text from transcript>", "confidence": <float>, "sentiment": <float>}}]
- No markdown, no prose outside JSON.

Title:
{title}

Description:
{description}

Transcript:
{content}
"""
    resp = _gemini_model.generate_content(prompt)
    text = (resp.text or "").strip()
    try:
        items = json.loads(text)
        if isinstance(items, list):
            cleaned = []
            for obj in items:
                if not isinstance(obj, dict):
                    continue
                name = str(obj.get("company", "")).strip()
                excerpt = str(obj.get("excerpt", "")).strip()
                conf = obj.get("confidence", None)
                sent = obj.get("sentiment", None)
                try:
                    conf_f = float(conf)
                except Exception:
                    conf_f = None
                try:
                    sent_f = float(sent)
                except Exception:
                    sent_f = None
                if name and excerpt:
                    cleaned.append(
                        {
                            "company": name,
                            "excerpt": excerpt,
                            "confidence": conf_f,
                            "sentiment": sent_f,
                        }
                    )
            if cleaned:
                return cleaned
    except Exception:
        pass

    # Fallback: basic split heuristic if JSON parsing fails
    parts = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    return [{"company": p, "excerpt": "", "confidence": None, "sentiment": None} for p in parts]


def fetch_and_save_metadata(kol_names):
    youtube_client = get_youtube_client()
    xlsx_paths = []
    for name in kol_names:
        print(f"Fetching metadata for KOL: {name}")
        channel_id = get_channel_id(youtube_client, name)
        if not channel_id:
            print(f"  Channel not found for {name}, skipping.")
            continue

        video_ids = fetch_video_ids(youtube_client, channel_id)
        if not video_ids:
            print(f"  No videos found for {name} in date range.")
            continue

        video_rows = fetch_video_details(youtube_client, video_ids, name)
        if not video_rows:
            print(f"  No video details fetched for {name}.")
            continue

        df = pd.DataFrame(video_rows)
        safe_name = sanitize_name(name)
        out_path = os.path.join(output_dir, f"youtube_videos_{safe_name}.xlsx")
        df.to_excel(out_path, index=False)
        xlsx_paths.append(out_path)
        print(f"  Saved {len(df)} rows to {out_path}")
    return xlsx_paths


def process_kol_file(xlsx_path: str):
    kol_slug = re.sub(r"^youtube_videos_", "", os.path.splitext(os.path.basename(xlsx_path))[0])
    try:
        df = pd.read_excel(xlsx_path)
    except Exception as e:
        print(f"Failed to read {xlsx_path}: {e}")
        return

    if df.empty:
        print(f"{xlsx_path} is empty, skipping.")
        return

    output_rows = []
    for idx, row in df.iterrows():
        print(f"[{kol_slug}] Processing video {idx + 1}/{len(df)}")
        video_id = str(row.get("video_id", "")).strip()
        if not video_id:
            print("  Missing video_id, skip.")
            continue

        transcript = download_transcript(video_id)
        if not transcript:
            print("  No transcript downloaded, skip company extraction.")
            continue

        title = row.get("title", "")
        description = row.get("description", "")
        company_mentions = find_company_llm(transcript, title, description)
        if not company_mentions:
            print("  No companies found.")
            continue

        for comp in company_mentions:
            output_rows.append(
                {
                    "channel_name": row.get("channel_name", kol_slug),
                    "video_id": video_id,
                    "publishedAt": row.get("publishedAt", ""),
                    "title": title,
                    "company": comp.get("company", ""),
                    "excerpt": comp.get("excerpt", ""),
                    "confidence": comp.get("confidence", None),
                    "sentiment": comp.get("sentiment", None),
                }
            )

    if output_rows:
        out_df = pd.DataFrame(output_rows)
        csv_path = os.path.join(output_dir, f"{kol_slug}_companies.csv")
        out_df.to_csv(csv_path, index=False)
        print(f"[{kol_slug}] Saved {len(out_df)} rows to {csv_path}")
    else:
        print(f"[{kol_slug}] No company data to save.")


def work():
    xlsx_paths = fetch_and_save_metadata(kol_list)
    if not xlsx_paths:
        print("No Excel files created, nothing to process.")
        return

    for path in xlsx_paths:
        process_kol_file(path)


if __name__ == "__main__":
    work()
