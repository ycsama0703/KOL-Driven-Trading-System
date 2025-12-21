import datetime as _dt
import re
from typing import Optional

from googleapiclient.discovery import build

# YouTube Data API key. Update this if you rotate keys.
API_KEY_YT = "AIzaSyCEMHaMsY1WhDToUh2T-81iEgR4sHbfVXE"


def _build_client():
    return build("youtube", "v3", developerKey=API_KEY_YT)


def _safe_channel_filename(channel_name: str) -> str:
    """Sanitize a channel name to match the file naming convention used in youtube_fetch.py."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", channel_name).strip("_")


def get_channel_id(channel_name: str, youtube=None) -> Optional[str]:
    """Get channelId by searching the channel name."""
    yt = youtube or _build_client()
    resp = (
        yt.search()
        .list(q=channel_name, type="channel", part="snippet", maxResults=1)
        .execute()
    )
    if not resp.get("items"):
        return None
    return resp["items"][0]["snippet"]["channelId"]


def count_videos_between_years(
    channel_name: str, start_year: int = 2022, end_year: int = 2024
) -> int:
    """
    Count how many videos a channel has between the given years (inclusive) via YouTube Data API.
    """
    youtube = _build_client()
    channel_id = get_channel_id(channel_name, youtube=youtube)
    if not channel_id:
        return 0

    start_dt = _dt.datetime(start_year, 1, 1, 0, 0, 0)
    end_dt = _dt.datetime(end_year, 12, 31, 23, 59, 59)
    published_after = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    published_before = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    count = 0
    page_token = None
    while True:
        resp = (
            youtube.search()
            .list(
                part="id",
                channelId=channel_id,
                publishedAfter=published_after,
                publishedBefore=published_before,
                maxResults=50,
                order="date",
                type="video",
                pageToken=page_token,
            )
            .execute()
        )

        items = resp.get("items", [])
        count += sum(1 for item in items if item.get("id", {}).get("videoId"))

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return count


if __name__ == "__main__":
    # Simple manual test
    kol_list = [
        # "Ale's World of Stocks",
        # "Tom Nash",
        # "MarketBeat",
        # "Fin Tek",
        # "Jerry Romine Stocks",
        # "Invest with Henry",
        # "Everything Money",

        # "Joseph Carlson",
        # "Ryan Scribner",
        # "Jay Fairbrother",
        "Daniel Pronk",
        # "Ale's World of Stocks",
        # "Tom Nash",
        # "MarketBeat",
        # "Fin Tek",
        # "Jerry Romine Stocks",
        # "Invest with Henry",
        # "Everything Money"
    ]

    channel = kol_list[0]
    total = count_videos_between_years(channel)
    print(f"{channel} has {total} videos between {2022} and {2024}.")
