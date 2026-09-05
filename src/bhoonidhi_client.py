"""Bhoonidhi (ISRO/NRSC) STAC-style API client -- auth + search only, no downloads."""
import os

import requests
from dotenv import load_dotenv

load_dotenv()

BHOONIDHI_USER_ID = os.environ.get("BHOONIDHI_USER_ID")
BHOONIDHI_PASSWORD = os.environ.get("BHOONIDHI_PASSWORD")

BASE_URL = "https://bhoonidhi-api.nrsc.gov.in"
TOKEN_URL = f"{BASE_URL}/auth/token"
COLLECTIONS_URL = f"{BASE_URL}/data/collections"
SEARCH_URL = f"{BASE_URL}/data/search"  # confirmed/adjusted after inspecting collections response


def require_credentials():
    if not BHOONIDHI_USER_ID or not BHOONIDHI_PASSWORD:
        raise RuntimeError(
            "BHOONIDHI_USER_ID / BHOONIDHI_PASSWORD not set. Add them to .env"
        )
    return BHOONIDHI_USER_ID, BHOONIDHI_PASSWORD


def get_token():
    user_id, password = require_credentials()
    resp = requests.post(
        TOKEN_URL,
        json={"userId": user_id, "password": password, "grant_type": "password"},
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    token = data.get("access_token") or data.get("token") or data.get("accessToken")
    if not token:
        raise RuntimeError(f"No token field found in auth response: {data}")
    return token, data


def get_collections(token):
    resp = requests.get(
        COLLECTIONS_URL,
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def search_items(token, collection_id, bbox, start_date, end_date, limit=50):
    """bbox as [minLon, minLat, maxLon, maxLat]; dates as YYYY-MM-DD."""
    payload = {
        "collections": [collection_id],
        "bbox": bbox,
        "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
        "limit": limit,
    }
    resp = requests.post(
        SEARCH_URL,
        json=payload,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        timeout=60,
    )
    return resp


if __name__ == "__main__":
    token, raw = get_token()
    print("AUTH OK. Token acquired (truncated):", token[:20], "...")

    cols = get_collections(token)
    print("\n--- /data/collections raw response (truncated to 4000 chars) ---")
    import json
    print(json.dumps(cols, indent=2)[:4000])
