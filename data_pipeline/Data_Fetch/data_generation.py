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

API_KEY_GEMINI = key3

# --- Config ---
channel_usernames = ["Daniel Pronk"]
# Joseph Carlson
# Ryan Scribner
# Jay Fairbrother
# Daniel Pronk
# Ale’s World of Stocks
# Tom Nash
# MarketBeat
# Fin Tek
# Jerry Romine Stocks
# Invest with Henry
# Everything Money
output_dir = r"C:\Users\Admin\Desktop\DFinTech\FT5007\SampleData"
excel_output_path = rf"{output_dir}\youtube_subtitles.xlsx"
gemini_model_name = "gemini-1.5-flash"
start_from = "2024-01-01T00:00:00Z"
os.makedirs(output_dir, exist_ok=True)

# --- Init APIs ---
youtube = build("youtube", "v3", developerKey=API_KEY_YT)
genai.configure(api_key=API_KEY_GEMINI)
model = genai.GenerativeModel(gemini_model_name)

# --- Functions ---
def get_channel_id(username):
    request = youtube.search().list(q=username, type="channel", part="snippet", maxResults=1)
    response = request.execute()
    if response["items"]:
        return response["items"][0]["snippet"]["channelId"]
    return None

def fetch_video_ids(channel_id):
    published_after = start_from
    video_ids = []
    next_page_token = None
    while True:
        request = youtube.search().list(
            part="id", channelId=channel_id, publishedAfter=published_after,
            maxResults=50, order="date", type="video", pageToken=next_page_token
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

def download_transcript(video_id, output_dir):
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

def split_text_into_chunks(text, max_chars=10000):
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

def summarize_long_transcript(title, description, transcript, model, chunk_size=10000):
    if not transcript.strip():
        return "No transcript available"

    chunks = split_text_into_chunks(transcript, max_chars=chunk_size)
    chunk_summaries = []

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
        try:
            response = model.generate_content(prompt)
            chunk_summaries.append(response.text.strip())
        except Exception as e:
            chunk_summaries.append(f"SUMMARY_ERROR: {e}")

    joined_chunks = "\n\n".join(chunk_summaries)
    final_prompt = f"""
                    You are a financial analyst. Based on the following summaries of different parts of a video transcript, produce a final concise summary that includes all key insights.

                    Summaries:
                    {joined_chunks}

                    Final summary:
                    """
    try:
        final_response = model.generate_content(final_prompt)
        return final_response.text.strip()
    except Exception as e:
        return f"SUMMARY_ERROR: {e}"


# --- Main ---
all_video_data = []

print("Starting YouTube → Transcript → Gemini summarization...\n")
for username in channel_usernames:
    channel_id = get_channel_id(username)
    if not channel_id:
        print(f"Could not find channel ID for {username}, skipping.")
        continue

    video_ids = fetch_video_ids(channel_id)
    if not video_ids:
        print(f"No videos found for {username} in 2024.")
        continue

    print(f"Found {len(video_ids)} videos for {username} in 2024.")
    video_details = fetch_video_details(video_ids, username)

    for video in tqdm(video_details, desc=f"Processing {username} videos", unit="video"):
        vid = video["video_id"]
        transcript = download_transcript(vid, output_dir)
        video["transcript"] = transcript
        if transcript:
            video["summary"] = summarize_long_transcript(video["title"], video["description"], transcript, model)
        else:
            video["summary"] = "No transcript available"
        all_video_data.append(video)

if all_video_data:
    df = pd.DataFrame(all_video_data)
    df.to_excel(excel_output_path, index=False)
    print(f"\n Saved all results to: {excel_output_path}")
else:
    print("\n No video data to save.")
