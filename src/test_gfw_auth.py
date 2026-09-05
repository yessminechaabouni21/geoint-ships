"""Step 2: confirm GFW API access works before building anything on top of it.

Run: python -m src.test_gfw_auth
"""
import sys

import requests

from src.config import GFW_API_BASE_URL, require_gfw_token


def test_auth():
    token = require_gfw_token()
    headers = {"Authorization": f"Bearer {token}"}

    url = f"{GFW_API_BASE_URL}/vessels/search"
    params = {
        "query": "IMO",
        "limit": 1,
        "datasets[0]": "public-global-vessel-identity:latest",
    }

    resp = requests.get(url, headers=headers, params=params, timeout=30)

    print(f"GET {url}")
    print(f"Status: {resp.status_code}")

    if resp.status_code == 200:
        data = resp.json()
        total = data.get("total")
        entries = len(data.get("entries", []))
        print(f"Auth OK. Vessel search returned total={total}, entries={entries}")
        return True

    print("Auth FAILED.")
    print(resp.text[:1000])
    return False


if __name__ == "__main__":
    ok = test_auth()
    sys.exit(0 if ok else 1)
