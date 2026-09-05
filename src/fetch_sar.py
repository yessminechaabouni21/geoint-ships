"""Step 3: pull SAR vessel detections from GFW for a bbox and date range."""
import pandas as pd

from src.config import SAR_DATASET, GFW_SPATIAL_RESOLUTION, GFW_TEMPORAL_RESOLUTION
from src.gfw_client import fetch_4wings_report


def fetch_sar_detections(bbox, start_date, end_date):
    """Return one row per SAR detection.

    Columns: lat, lon, timestamp (from entryTimestamp), mmsi, ais_matched
    (mmsi non-empty), plus identity fields when matched.
    """
    rows = fetch_4wings_report(
        dataset=SAR_DATASET,
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        spatial_resolution=GFW_SPATIAL_RESOLUTION,
        temporal_resolution=GFW_TEMPORAL_RESOLUTION,
    )

    if not rows:
        return pd.DataFrame(columns=[
            "sar_id", "timestamp", "lat", "lon", "mmsi", "ais_matched",
            "ship_name", "flag", "vessel_type",
        ])

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["entryTimestamp"], utc=True, errors="coerce")
    df["mmsi"] = df["mmsi"].replace("", pd.NA)
    df["ais_matched"] = df["mmsi"].notna()
    df["sar_id"] = df.index.astype(str)

    df = df.rename(columns={
        "shipName": "ship_name",
        "flag": "flag",
        "vesselType": "vessel_type",
    })

    keep = ["sar_id", "timestamp", "lat", "lon", "mmsi", "ais_matched",
            "ship_name", "flag", "vessel_type"]
    return df[keep].sort_values("timestamp").reset_index(drop=True)


if __name__ == "__main__":
    from src.config import TEST_BBOX, TEST_START_DATE, TEST_END_DATE, RUN_LABEL

    df = fetch_sar_detections(TEST_BBOX, TEST_START_DATE, TEST_END_DATE)
    print(f"Fetched {len(df)} SAR detections for {RUN_LABEL}")
    print(df.head(10))
    print(f"Matched: {df['ais_matched'].sum()}, Unmatched (dark): {(~df['ais_matched']).sum()}")

    out_path = f"data/raw/{RUN_LABEL}_sar_detections.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
