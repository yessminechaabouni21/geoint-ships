"""Step 5: interpolate AIS position to an arbitrary timestamp per vessel.

Input is GFW's hourly grid-cell presence bins (see src/fetch_ais.py caveat),
not raw pings. Each bin's cell centroid + midpoint timestamp is treated as one
coarse position sample; we linearly interpolate between the two nearest
samples (by time) that bracket the query timestamp.
"""
import numpy as np
import pandas as pd


def _dedupe_ais(ais_df):
    return (
        ais_df.drop_duplicates(subset=["mmsi", "timestamp", "lat", "lon"])
        .sort_values(["mmsi", "timestamp"])
        .reset_index(drop=True)
    )


def build_ais_tracks(ais_df):
    """Group deduped AIS position samples by MMSI, sorted by time.

    Returns dict[mmsi] -> DataFrame sorted by timestamp.
    """
    ais_df = _dedupe_ais(ais_df)
    return {mmsi: g.reset_index(drop=True) for mmsi, g in ais_df.groupby("mmsi")}


def interpolate_position(track_df, query_time):
    """Linearly interpolate a vessel's position at query_time.

    track_df: single-vessel DataFrame with columns timestamp, lat, lon,
    sorted ascending by timestamp (one row per hourly grid-cell bin).
    query_time: a pandas Timestamp (tz-aware, UTC).

    Returns (lat, lon, time_gap_seconds) or None if track_df is empty or
    query_time falls outside the track's time span.
    """
    if track_df.empty:
        return None

    times = track_df["timestamp"].values
    query_ns = np.datetime64(query_time.tz_convert("UTC").tz_localize(None))

    if query_ns < times[0] or query_ns > times[-1]:
        return None

    idx_after = np.searchsorted(times, query_ns, side="left")

    if times[idx_after] == query_ns:
        row = track_df.iloc[idx_after]
        return float(row["lat"]), float(row["lon"]), 0.0

    idx_before = idx_after - 1
    t0, t1 = times[idx_before], times[idx_after]
    row0, row1 = track_df.iloc[idx_before], track_df.iloc[idx_after]

    span = (t1 - t0) / np.timedelta64(1, "s")
    if span == 0:
        frac = 0.0
    else:
        frac = ((query_ns - t0) / np.timedelta64(1, "s")) / span

    lat = row0["lat"] + frac * (row1["lat"] - row0["lat"])
    lon = row0["lon"] + frac * (row1["lon"] - row0["lon"])

    gap_seconds = min(
        (query_ns - t0) / np.timedelta64(1, "s"),
        (t1 - query_ns) / np.timedelta64(1, "s"),
    )
    return float(lat), float(lon), float(gap_seconds)
