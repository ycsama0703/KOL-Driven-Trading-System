import pandas as pd
import time
import google.generativeai as genai
from tqdm import tqdm

# --- Config ---
input_excel_path = r"C:\Users\Admin\Desktop\DFinTech\FT5007\SampleData\youtube_subtitles_Daniel_Pronk.xlsx"
output_excel_path = input_excel_path

API_KEY_GEMINI = key3
gemini_model_name = "gemini-1.5-flash"

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


# --- Load and Process ---
df = pd.read_excel(input_excel_path)

# Identify rows needing update
error_rows = df[df["summary"].apply(is_error_summary)]

if error_rows.empty:
    print("✅ No missing or error summaries found.")
else:
    print(f"🔁 Retrying {len(error_rows)} missing/error summaries...\n")
    for idx in tqdm(error_rows.index, desc="Regenerating summaries"):
        row = df.loc[idx]
        transcript = str(row["transcript"])
        if transcript.strip():
            new_summary = summarize_long_transcript(row["title"], row["description"], transcript, model)
            df.at[idx, "summary"] = new_summary
            time.sleep(1)  # Gemini rate limit protection
        else:
            df.at[idx, "summary"] = "No transcript available"

    # Save result
    df.to_excel(output_excel_path, index=False)
    print(f"\n✅ Updated summaries saved to: {output_excel_path}")
