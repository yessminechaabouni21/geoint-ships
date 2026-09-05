"""Steps 6-7: match each SAR detection to the nearest AIS position and classify.

Classification is distance-tiered, with NO distance cap on the search itself
-- every SAR detection gets a nearest-AIS distance unless there is truly zero
AIS activity anywhere in the bbox during that hour. This replaces an earlier
version that capped the search radius and dumped anything beyond it into an
"unmatched" bucket; a diagnostic (src/check_unbounded_nearest.py) showed that
bucket was hiding far-away AIS activity (mean nearest distance ~9-25km, not
"no data") -- i.e. severe spoofing was being misclassified as AIS silence.
"""
import numpy as np
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd

from src.interpolate import build_ais_tracks, interpolate_position

EARTH_RADIUS_KM = 6371.0088

MATCHED_THRESHOLD_KM = 1.5
SPOOFED_THRESHOLD_KM = 20.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _nearest_in_hour_bin(sar_lat, sar_lon, bin_ais):
    """Nearest AIS position among ALL vessels in bin_ais, unbounded distance.

    Returns (mmsi, lat, lon, distance_km) or None if bin_ais is empty.
    """
    if bin_ais.empty:
        return None
    d = haversine_km(sar_lat, sar_lon, bin_ais["lat"].values, bin_ais["lon"].values)
    idx = np.argmin(d)
    row = bin_ais.iloc[idx]
    return row["mmsi"], float(row["lat"]), float(row["lon"]), float(d[idx])


def match_and_classify(sar_df, ais_df, matched_threshold_km=MATCHED_THRESHOLD_KM,
                        spoofed_threshold_km=SPOOFED_THRESHOLD_KM):
    """For each SAR detection, find the nearest AIS position (no distance cap)
    and classify by distance tier.

    Matching strategy per SAR detection:
      1. If the detection has an mmsi and that vessel has an AIS track
         spanning the detection's timestamp, use the time-interpolated
         position on that vessel's own track (identity-based match).
      2. Otherwise (no mmsi, or no usable track), fall back to the nearest
         AIS position from ANY vessel in the same hourly bin, unbounded.
      3. Only if there is zero AIS activity anywhere in the bbox during that
         hour does the detection get no distance at all.

    Classification (pure distance tiers, no time cap):
      - "matched":        distance_km <= matched_threshold_km
      - "discrepant":     matched_threshold_km < distance_km <= spoofed_threshold_km
                           (ambiguous middle -- consistent with either hourly-bin
                           noise or subtle spoofing; flagged inconclusive by design)
      - "likely_spoofed": distance_km > spoofed_threshold_km
      - "no_ais_activity": no AIS position exists anywhere in the bbox for
                           that hour bin at all (true silence)

    Returns a DataFrame with one row per SAR detection.
    """
    tracks = build_ais_tracks(ais_df)

    ais_df = ais_df.copy()
    ais_df["hour_bin"] = ais_df["timestamp"].dt.floor("h")

    results = []
    for _, sar_row in sar_df.iterrows():
        base = {
            "sar_id": sar_row["sar_id"],
            "sar_timestamp": sar_row["timestamp"],
            "sar_lat": sar_row["lat"],
            "sar_lon": sar_row["lon"],
            "mmsi": sar_row["mmsi"],
            "ship_name": sar_row.get("ship_name"),
            "vessel_type": sar_row.get("vessel_type"),
            "flag": sar_row.get("flag"),
        }

        ais_lat = ais_lon = distance_km = time_gap_seconds = matched_mmsi = None

        if pd.notna(sar_row["mmsi"]):
            track = tracks.get(sar_row["mmsi"])
            interp = interpolate_position(track, sar_row["timestamp"]) if track is not None else None
            if interp is not None:
                ais_lat, ais_lon, time_gap_seconds = interp
                distance_km = haversine_km(sar_row["lat"], sar_row["lon"], ais_lat, ais_lon)
                matched_mmsi = sar_row["mmsi"]

        if distance_km is None:
            hour_bin = sar_row["timestamp"].floor("h")
            bin_ais = ais_df[ais_df["hour_bin"] == hour_bin]
            nearest = _nearest_in_hour_bin(sar_row["lat"], sar_row["lon"], bin_ais)
            if nearest is not None:
                matched_mmsi, ais_lat, ais_lon, distance_km = nearest

        if distance_km is None:
            classification = "no_ais_activity"
        elif distance_km <= matched_threshold_km:
            classification = "matched"
        elif distance_km <= spoofed_threshold_km:
            classification = "discrepant"
        else:
            classification = "likely_spoofed"

        results.append({
            **base,
            "ais_lat": ais_lat, "ais_lon": ais_lon,
            "matched_mmsi": matched_mmsi,
            "distance_km": distance_km, "time_gap_seconds": time_gap_seconds,
            "classification": classification,
        })

    return pd.DataFrame(results)


def to_geodataframe(match_df):
    geometry = [Point(xy) for xy in zip(match_df["sar_lon"], match_df["sar_lat"])]
    return gpd.GeoDataFrame(match_df, geometry=geometry, crs="EPSG:4326")


if __name__ == "__main__":
    from src.config import TEST_BBOX, TEST_START_DATE, TEST_END_DATE, RUN_LABEL
    from src.fetch_sar import fetch_sar_detections
    from src.fetch_ais import fetch_ais_positions

    sar_df = fetch_sar_detections(TEST_BBOX, TEST_START_DATE, TEST_END_DATE)
    ais_df = fetch_ais_positions(TEST_BBOX, TEST_START_DATE, TEST_END_DATE)

    result = match_and_classify(sar_df, ais_df)
    print(result["classification"].value_counts())

    out_path = f"data/processed/{RUN_LABEL}_classified.csv"
    result.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
