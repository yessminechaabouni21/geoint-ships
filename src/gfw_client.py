"""Shared helper for calling GFW's 4wings /report endpoint."""
import requests

from src.config import GFW_API_BASE_URL, require_gfw_token


def bbox_to_geojson_polygon(min_lat, max_lat, min_lon, max_lon):
    return {
        "type": "Polygon",
        "coordinates": [[
            [min_lon, min_lat],
            [max_lon, min_lat],
            [max_lon, max_lat],
            [min_lon, max_lat],
            [min_lon, min_lat],
        ]],
    }


def fetch_4wings_report(dataset, bbox, start_date, end_date,
                         spatial_resolution="HIGH", temporal_resolution="HOURLY"):
    """Call the 4wings report endpoint for one dataset over a bbox/date range.

    Returns the raw list of entry dicts (already unwrapped from the
    dataset-keyed envelope GFW wraps them in).
    """
    token = require_gfw_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    url = f"{GFW_API_BASE_URL}/4wings/report"
    params = [
        ("spatial-resolution", spatial_resolution),
        ("temporal-resolution", temporal_resolution),
        ("format", "JSON"),
        ("datasets[0]", dataset),
        ("date-range", f"{start_date},{end_date}"),
    ]
    body = {"geojson": bbox_to_geojson_polygon(**bbox)}

    resp = requests.post(url, headers=headers, params=params, json=body, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for entry in data.get("entries", []):
        for dataset_key, dataset_rows in entry.items():
            if dataset_rows:
                rows.extend(dataset_rows)
    return rows
