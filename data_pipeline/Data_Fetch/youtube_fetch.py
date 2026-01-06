import os
import re
import subprocess
import time
import pandas as pd
from datetime import datetime, timedelta
from googleapiclient.discovery import build

# --- API Key & config ---

output_dir = "./youtube_kol_outputs"
os.makedirs(output_dir, exist_ok=True)

# --- Date window: last 3 years up to today ---
end_date = datetime.now()
start_date = end_date - timedelta(days=365 * 3)
published_after = start_date.strftime("%Y-%m-%dT00:00:00Z")
published_before = end_date.strftime("%Y-%m-%dT23:59:59Z")

# --- Target KOL channels ---
kol_list = [
    # "Ale's World of Stocks",
    # "Tom Nash",
    # "MarketBeat",
    # "Fin Tek",
    # "Jerry Romine Stocks",
    # "Invest with Henry",
    "Everything Money",
]

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
            publishedAfter=published_after,
            publishedBefore=published_before,
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
        time.sleep(0.4)
    return video_ids


def fetch_video_details(video_ids, channel_name):
    details = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i : i + 50]
        resp = youtube.videos().list(
            part="snippet,statistics", id=",".join(batch)
        ).execute()
        for item in resp["items"]:
            snip = item["snippet"]
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


if __name__ == "__main__":
    for name in kol_list:
        print(f"Fetching videos for: {name}")
        cid = get_channel_id(name)
        if not cid:
            print("  Channel not found, skip.")
            continue
        vids = fetch_video_ids(cid)
        if not vids:
            print("  No videos in range.")
            continue
        video_rows = fetch_video_details(vids, name)

        for video in video_rows:
            vid = video.get("video_id") or video.get("id")
            video["transcript"] = download_transcript(vid) if vid else ""

        if video_rows:
            df = pd.DataFrame(video_rows)
            safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
            out_path = os.path.join(output_dir, f"{safe_name}.xlsx")
            df.to_excel(out_path, index=False)
            print(f"  Saved {len(df)} rows to {out_path}")
        else:
            print("  No data collected for this channel.")
