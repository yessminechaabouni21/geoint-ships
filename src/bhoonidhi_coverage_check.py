"""Coverage-check search (no downloads) over the Jamnagar/Vadinar, Gujarat AOI.

Confirmed API shape (probed live against /data/collections and /data/search):
  - Auth: POST /auth/token {"userId","password","grant_type":"password"} -> access_token
  - Collections: GET /data/collections -> {"collections":[{id, extent.temporal.interval, ...}]}
  - Search: POST /data/search
      {"collections": [id], "datetime": "START/END" (ISO8601 interval), "bbox": [minLon,minLat,maxLon,maxLat]}
    NOTE: minLat/maxLat/minLon/maxLon and startDate/endDate/fromDate/etc. keys are
    silently IGNORED by this endpoint (returns latest 500 items unfiltered). Only
    "datetime" (STAC interval string) and "bbox" (array) actually filter.
  - Each returned feature's properties has "Online": "Y"/"N".
"""
import requests

from src.bhoonidhi_client import get_token, SEARCH_URL

AOI_BBOX = [69.0, 22.0, 70.0, 23.0]  # Jamnagar/Vadinar, Gujarat

COLLECTIONS = [
    "EOS-04_SAR-MRS_L2B",
    "Sentinel-1A_SAR-IW_GRD",
    "Sentinel-1A_SAR-IW_SLC",
]

# Three non-overlapping 5-day windows, all within the last 3 months of today
# (2026-09-05) AND within the tightest confirmed archive extent among the
# tested collections (Sentinel-1A_SAR-IW_GRD/SLC end 2026-06-30T00:00:00Z).
WINDOWS = [
    ("2026-06-05", "2026-06-09"),
    ("2026-06-13", "2026-06-17"),
    ("2026-06-21", "2026-06-25"),
]


def search(token, collection_id, start_date, end_date, bbox):
    payload = {
        "collections": [collection_id],
        "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
        "bbox": bbox,
    }
    resp = requests.post(
        SEARCH_URL,
        json=payload,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        timeout=60,
    )
    if resp.status_code == 404:
        # This API returns HTTP 404 with an ErrorCode body for "no results",
        # not a routing error -- treat as zero items.
        body = resp.json()
        if body.get("ErrorCode") == "404":
            return 0, 0, 0
        raise RuntimeError(f"Unexpected 404 for {collection_id} {start_date}/{end_date}: {body}")
    resp.raise_for_status()
    data = resp.json()
    features = data.get("features", [])
    online = sum(1 for f in features if f.get("properties", {}).get("Online") == "Y")
    offline = len(features) - online
    return len(features), online, offline


if __name__ == "__main__":
    token, _ = get_token()

    rows = []
    for collection_id in COLLECTIONS:
        for start_date, end_date in WINDOWS:
            count, online, offline = search(token, collection_id, start_date, end_date, AOI_BBOX)
            rows.append((collection_id, f"{start_date}/{end_date}", count, online, offline))

    header = f"{'Collection':<28} {'Window':<24} {'Items':>6} {'Online':>7} {'Offline':>8}"
    print(header)
    print("-" * len(header))
    for collection_id, window, count, online, offline in rows:
        print(f"{collection_id:<28} {window:<24} {count:>6} {online:>7} {offline:>8}")

    total = sum(r[2] for r in rows)
    nonzero = sum(1 for r in rows if r[2] > 0)
    print(f"\nTotal items across {len(rows)} combinations: {total}")
    print(f"Non-zero combinations: {nonzero} / {len(rows)}")
