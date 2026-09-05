"""Step 4: pull AIS presence data from GFW for a bbox and date range.

Caveat: GFW's public API exposes AIS as hourly grid-cell PRESENCE bins, not
raw pings (see src/config.py). Each row here is one vessel's presence in one
~1.1km cell during one hour, not a discrete position report. Downstream
interpolation (src/interpolate.py) treats the cell centroid and the bin's
entry/exit timestamps as a coarse position sample.
"""
import pandas as pd

from src.config import AIS_PRESENCE_DATASET, GFW_SPATIAL_RESOLUTION, GFW_TEMPORAL_RESOLUTION
from src.gfw_client import fetch_4wings_report


def fetch_ais_positions(bbox, start_date, end_date):
    """Return one row per vessel per hourly grid-cell presence bin.

    Columns: mmsi, timestamp, lat, lon, ship_name, flag, vessel_type.

    IMPORTANT: the "timestamp" here comes from each row's "date" field, which
    is the actual per-row HOURLY bucket (e.g. "2025-10-05 02:00"). Do NOT use
    "entryTimestamp"/"exitTimestamp" for positioning -- for this report shape
    (no group-by, HIGH spatial / HOURLY temporal resolution) those two fields
    are constant across every row for a vessel, equal to the overall query
    date-range bounds, not that cell's actual dwell window. Using them for
    position lookup previously caused every cell to appear to "contain" any
    query timestamp in the whole date range, producing wildly wrong matches.
    """
    rows = fetch_4wings_report(
        dataset=AIS_PRESENCE_DATASET,
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        spatial_resolution=GFW_SPATIAL_RESOLUTION,
        temporal_resolution=GFW_TEMPORAL_RESOLUTION,
    )

    if not rows:
        return pd.DataFrame(columns=[
            "mmsi", "timestamp", "lat", "lon", "ship_name", "flag", "vessel_type",
        ])

    df = pd.DataFrame(rows)
    df = df[df["mmsi"].astype(str).str.len() > 0].copy()

    df["timestamp"] = pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M", utc=True, errors="coerce")

    df = df.rename(columns={"shipName": "ship_name", "vesselType": "vessel_type"})

    keep = ["mmsi", "timestamp", "lat", "lon", "ship_name", "flag", "vessel_type"]
    return df[keep].dropna(subset=["timestamp"]).sort_values(["mmsi", "timestamp"]).reset_index(drop=True)


if __name__ == "__main__":
    from src.config import TEST_BBOX, TEST_START_DATE, TEST_END_DATE, RUN_LABEL

    df = fetch_ais_positions(TEST_BBOX, TEST_START_DATE, TEST_END_DATE)
    print(f"Fetched {len(df)} AIS presence bins for {RUN_LABEL}")
    print(f"Unique vessels (MMSI): {df['mmsi'].nunique()}")
    print(df.head(10))

    out_path = f"data/raw/{RUN_LABEL}_ais_positions.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
