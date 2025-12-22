import os
import pandas as pd
import time
import google.generativeai as genai
from tqdm import tqdm

# --- Config ---
input_dir = r"C:\Users\Admin\Desktop\DFinTech\FT5007\youtube_kol_outputs"
API_KEY_GEMINI = 
gemini_model_name = "gemini-2.5-flash"

# --- Init Gemini ---
genai.configure(api_key=API_KEY_GEMINI)
model = genai.GenerativeModel(gemini_model_name)

def is_error_summary(text):
    if not isinstance(text, str):
        return True
    return (
        "SUMMARY_ERROR" in text
        or "No transcript available" in text
        or "exceeded your current quota" in text
        or "Please try again later" in text
        or "Prompt blocked" in text
    )

def split_text_into_chunks(text, max_chars=30000):
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

def summarize_long_transcript(title, description, transcript, model, chunk_size=30000, max_retries=3):
    if not transcript.strip():
        return "No transcript available"

    for attempt in range(max_retries):
        try:
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
                response = model.generate_content(prompt)
                chunk_summaries.append(response.text.strip())

            joined_chunks = "\n\n".join(chunk_summaries)
            final_prompt = f"""
You are a financial analyst. Based on the following summaries of different parts of a video transcript, produce a final concise summary that includes all key insights.

Summaries:
{joined_chunks}

Final summary:
"""
            final_response = model.generate_content(final_prompt)
            return final_response.text.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                return f"SUMMARY_ERROR: {e}"
            time.sleep(10)

# --- Process all Excel files ---
excel_files = [f for f in os.listdir(input_dir) if f.endswith(".xlsx")]
for filename in excel_files:
    file_path = os.path.join(input_dir, filename)
    df = pd.read_excel(file_path)

    error_rows = df[df["summary"].apply(is_error_summary)]

    if error_rows.empty:
        print(f"No error summaries found in: {filename}")
        continue

    print(f"Recovering {len(error_rows)} rows in: {filename}")
    for idx in tqdm(error_rows.index, desc=f"Recovering {filename}"):
        row = df.loc[idx]
        transcript = str(row["transcript"])
        if transcript.strip():
            summary = summarize_long_transcript(row["title"], row["description"], transcript, model)
            if summary.startswith("SUMMARY_ERROR"):
                print(f"Aborted: Persistent error during {filename}. Saving current progress...")
                df.to_excel(file_path, index=False)
                raise SystemExit(f"Stopped due to error:\n{summary}")
            df.at[idx, "summary"] = summary
            time.sleep(1)
        else:
            df.at[idx, "summary"] = "No transcript available"

    df.to_excel(file_path, index=False)
    print(f"Saved recovered summaries to: {file_path}")
