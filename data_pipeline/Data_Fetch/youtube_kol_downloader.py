import os
import re
import time
import pandas as pd
import subprocess
from datetime import datetime
from googleapiclient.discovery import build
import google.generativeai as genai
from tqdm import tqdm

# --- API Keys ---

gemini_model_name = "gemini-2.5-flash"

# --- Config ---
output_dir = r"./youtube_kol_outputs"
start_date = "2022-01-01"  # YYYY-MM-DD
end_date = "2024-12-31"
os.makedirs(output_dir, exist_ok=True)

# --- KOL List ---
kol_list = [
    # "Joseph Carlson",
    # "Ryan Scribner",
    # "Jay Fairbrother",
    # "Daniel Pronk",
    # "Ale's World of Stocks",
    # "Tom Nash",
    "MarketBeat",
    # "Fin Tek",
    # "Jerry Romine Stocks",
    "Invest with Henry",
    "Everything Money"
]

# --- Init APIs ---
youtube = build("youtube", "v3", developerKey=API_KEY_YT)
genai.configure(api_key=API_KEY_GEMINI)
model = genai.GenerativeModel(gemini_model_name)

# --- Functions ---
def split_text_into_chunks(text, max_chars=30000):
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

def summarize_long_transcript_with_retries(title, description, transcript, model, chunk_size=30000, max_retries=3):
    if not transcript.strip():
        return "No transcript available", False

    chunks = split_text_into_chunks(transcript, max_chars=chunk_size)
    chunk_summaries = []
    retry_error = None

    for i, chunk in enumerate(chunks):
        prompt = f"""
You are a financial content analyst. Summarize the following part of a YouTube transcript.

Title: {title}
Description: {description}
Transcript Part {i+1}/{len(chunks)}:
{chunk}

Focus on:
1. Investment opinions or recommendations.
2. The reasoning or evidence behind those opinions.
3. Any noteworthy market commentary.

Provide a concise summary of this part.
"""
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                chunk_summaries.append(response.text.strip())
                time.sleep(2)
                break
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg and "quota" in error_msg:
                    retry_error = f"SUMMARY_ERROR: {e}"
                    delay = 10 + attempt * 20
                    print(f"429 Rate Limit. Retry {attempt + 1}/{max_retries} after {delay} seconds...")
                    time.sleep(delay)
                else:
                    chunk_summaries.append(f"SUMMARY_ERROR: {e}")
                    break
        else:
            return retry_error, True

    joined_chunks = "\n\n".join(chunk_summaries)
    final_prompt = f"""
You are a financial analyst. Based on the following summaries of different parts of a video transcript, produce a final concise summary that includes all key insights.

Summaries:
{joined_chunks}

Final summary:
"""
    for attempt in range(max_retries):
        try:
            response = model.generate_content(final_prompt)
            time.sleep(2)
            return response.text.strip(), False
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg and "quota" in error_msg:
                retry_error = f"SUMMARY_ERROR: {e}"
                delay = 10 + attempt * 20
                print(f"429 Rate Limit (final summary). Retry {attempt + 1}/{max_retries} after {delay} seconds...")
                time.sleep(delay)
            else:
                return f"SUMMARY_ERROR: {e}", False
    return retry_error, True

def get_channel_id(username):
    request = youtube.search().list(q=username, type="channel", part="snippet", maxResults=1)
    response = request.execute()
    if response["items"]:
        return response["items"][0]["snippet"]["channelId"]
    return None

def fetch_video_ids(channel_id, start_date, end_date):
    video_ids = []
    next_page_token = None
    published_after = start_date + "T00:00:00Z"
    published_before = end_date + "T23:59:59Z"

    while True:
        request = youtube.search().list(
            part="id", channelId=channel_id,
            publishedAfter=published_after,
            publishedBefore=published_before,
            maxResults=50, order="date", type="video",
            pageToken=next_page_token
        )
        response = request.execute()
        video_ids += [item["id"]["videoId"] for item in response["items"] if "videoId" in item["id"]]
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
        time.sleep(0.5)
    return video_ids

def fetch_video_details(video_ids, channel_name):
    all_details = []
    for i in range(0, len(video_ids), 50):
        batch_ids = video_ids[i:i+50]
        request = youtube.videos().list(part="snippet", id=",".join(batch_ids))
        response = request.execute()
        for item in response["items"]:
            snippet = item["snippet"]
            all_details.append({
                "video_id": item["id"],
                "channel_name": channel_name,
                "publishedAt": snippet.get("publishedAt", ""),
                "title": snippet.get("title", ""),
                "description": snippet.get("description", "")
            })
        time.sleep(0.5)
    return all_details

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

# --- Main ---
print("Starting YouTube → Transcript → Gemini summarization...")

for username in kol_list:
    print(f"\nProcessing channel: {username}")
    channel_id = get_channel_id(username)
    if not channel_id:
        print(f"Could not find channel ID for {username}, skipping.")
        continue

    video_ids = fetch_video_ids(channel_id, start_date, end_date)
    if not video_ids:
        print(f"No videos found for {username} in the selected period.")
        continue

    print(f"Found {len(video_ids)} videos for {username}.")
    video_details = fetch_video_details(video_ids, username)
    all_video_data = []

    for idx, video in enumerate(tqdm(video_details, desc=f"Summarizing {username}", unit="video")):
        vid = video["video_id"]
        transcript = download_transcript(vid)
        video["transcript"] = transcript
        if transcript:
            summary, stop_flag = summarize_long_transcript_with_retries(video["title"], video["description"], transcript, model)
            video["summary"] = summary
            if stop_flag:
                print("Quota limit hit. Stopping early.")
                for rest in video_details[idx+1:]:
                    rest["transcript"] = ""
                    rest["summary"] = summary
                    all_video_data.append(rest)
                break
        else:
            video["summary"] = "No transcript available"
        all_video_data.append(video)

    if all_video_data:
        df = pd.DataFrame(all_video_data)
        safe_name = username.replace("’", "").replace("'", "").replace(" ", "_")
        output_path = os.path.join(output_dir, f"youtube_subtitles_{safe_name}.xlsx")
        df.to_excel(output_path, index=False)
        print(f"Saved {len(df)} entries to: {output_path}")
    else:
        print(f"No data collected for {username}, skipping file write.")
