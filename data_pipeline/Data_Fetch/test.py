import os
import re
import time
import json
import pandas as pd
import subprocess
import google.generativeai as genai
from googleapiclient.discovery import build

# --- API Key & date window ---

PUBLISHED_AFTER = "2022-01-01T00:00:00Z"
PUBLISHED_BEFORE = "2024-12-31T23:59:59Z"

GEMINI_MODEL = "gemini-2.5-flash"

_gemini_model = None

# --- Target KOL channels ---
kol_list = [
    "Ale's World of Stocks",
    # "Tom Nash",
    # "MarketBeat",
    # "Fin Tek",
    # "Jerry Romine Stocks",
    # "Invest with Henry",
    # "Everything Money",
]

output_dir = "./youtube_kol_outputs"
os.makedirs(output_dir, exist_ok=True)

youtube = build("youtube", "v3", developerKey=API_KEY_YT)


def get_channel_id(username: str):
    resp = youtube.search().list(
        q=username, type="channel", part="snippet", maxResults=1
    ).execute()
    return resp["items"][0]["snippet"]["channelId"] if resp["items"] else None


def fetch_video_ids(channel_id: str):
    video_ids, token = [], None
    while True:
        resp = youtube.search().list(
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
            for item in resp["items"]
            if "videoId" in item["id"]
        ]
        token = resp.get("nextPageToken")
        if not token:
            break
        time.sleep(0.4)  # keep within rate limits
    return video_ids


def fetch_video_details(video_ids, channel_name):
    details = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i : i + 50]
        resp = youtube.videos().list(
            part="snippet", id=",".join(batch)
        ).execute()
        for item in resp["items"]:
            snip = item["snippet"]
            details.append(
                {
                    "channel_name": channel_name,
                    "video_id": item.get("id", ""),
                    "publishedAt": snip.get("publishedAt", ""),
                    "title": snip.get("title", ""),
                    "description": snip.get("description", ""),
                }
            )
        time.sleep(0.3)
    return details


def download_transcript(video_id):
    vtt_path = os.path.join(output_dir, f"{video_id}.en.vtt")
    try:
        cmd = [
            "yt-dlp", "--write-auto-sub", "--sub-lang", "en", "--skip-download",
            "-o", os.path.join(output_dir, "%(id)s.%(ext)s"),
            f"https://www.youtube.com/watch?v={video_id}"
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=False)

        if not os.path.exists(vtt_path):
            vtts = [f for f in os.listdir(output_dir) if f.startswith(video_id) and f.endswith(".vtt")]
            if vtts:
                vtt_path = os.path.join(output_dir, vtts[0])
            else:
                return ""

        text_lines = []
        with open(vtt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("WEBVTT") or re.match(r"^\d\d:\d\d:\d\d\.\d\d\d -->", line):
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
                except:
                    pass

def read_first_video_id_from_outputs():
    """
    Pick the first Excel file in youtube_kol_outputs and return its first video_id.
    """
    xlsx_files = sorted(
        f for f in os.listdir(output_dir) if f.lower().endswith(".xlsx")
    )
    if not xlsx_files:
        print("No Excel files found in youtube_kol_outputs.")
        return None

    first_file = os.path.join(output_dir, xlsx_files[0])
    df = pd.read_excel(first_file)
    for col in ("video_id", "id", "videoId"):
        if col in df.columns:
            series = df[col].dropna()
            if not series.empty:
                vid = str(series.iloc[0])
                print(f"Using video_id from {first_file}: {vid}")
                return vid
    print(f"No video_id column found in {first_file}.")
    return None


def download_first_transcript_from_outputs():
    vid = read_first_video_id_from_outputs()
    if not vid:
        return
    transcript = download_transcript(vid) or ""
    out_path = os.path.join(output_dir, f"{vid}_transcript.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    print(f"Saved transcript to {out_path} (len={len(transcript)})")


def find_company_llm(content, title: str = "", description: str = ""):
    """
    Use Gemini to extract distinct company names with a short supporting excerpt.
    Considers title/transcript (ignore ads/promos). Returns [{"company": str, "excerpt": str}].
    """
    global _gemini_model
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    if _gemini_model is None:
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_model = genai.GenerativeModel(GEMINI_MODEL)

    prompt = f"""
You are extracting company mentions.
- Use the title and transcript to find distinct company names.
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


if __name__ == "__main__":
    # for name in kol_list:
    #     print(f"Fetching videos for: {name}")
    #     cid = get_channel_id(name)
    #     if not cid:
    #         print("  Channel not found, skip.")
    #         continue
    #
    #     vids = fetch_video_ids(cid)
    #     if not vids:
    #         print("  No videos in range.")
    #         continue
    #
    #     video_rows = fetch_video_details(vids, name)
    #     if video_rows:
    #         df = pd.DataFrame(video_rows)
    #         safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    #         out_path = os.path.join(output_dir, f"{safe_name}.xlsx")
    #         df.to_excel(out_path, index=False)
    #         print(f"  Saved {len(df)} rows to {out_path}")
    #     else:
    #         print("  No data collected for this channel.")
    transcript_path = ""
    for f in os.listdir(output_dir):
        if f.endswith(".txt"):
            transcript_path = os.path.join(output_dir, f)
            break
    if not transcript_path:
        raise RuntimeError("No transcript .txt found in youtube_kol_outputs.")
    with open(transcript_path, 'r', encoding='utf-8') as f:
        content = f.read()

    title = "Warren Buffett Just Bought These NEW Stocks Instead of Apple! 👀"

    print("\nGemini companies with excerpts:")
    company_llm = find_company_llm(content, title)
    for item in company_llm:
        print(item)

    out_json = os.path.join(output_dir, "company_mentions1.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(company_llm, f, ensure_ascii=False, indent=2)
    print(f"\nSaved company mentions to {out_json}")
