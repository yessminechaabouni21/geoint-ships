"""Standalone check (does not touch match.py): are AIS-claimed positions
physically implausible -- on land, or suspiciously close to a landmark that
featured in documented spoofing reports -- during the Hormuz crisis window
vs. a control window?

This sidesteps the matched/discrepant/unmatched taxonomy entirely: it looks
only at raw AIS hourly-bin positions (src/fetch_ais.py output) and asks
whether the claimed position itself makes sense, independent of whether a
SAR detection was ever associated with it.

Land check: uses the `global_land_mask` package (GSHHG shoreline data,
sub-km resolution) rather than geopandas' `naturalearth_lowres`, which
GeoPandas >=1.0 removed. GSHHG is a finer-resolution shoreline product than
Natural Earth 1:10m, so this is a strict upgrade for this purpose, not a
downgrade.
"""
import pandas as pd
from global_land_mask import globe

from src.match import haversine_km

LANDMARKS = {
    "Bushehr nuclear plant, Iran": (28.9669, 50.8878),
    "Bandar Abbas Airport (BND), Iran": (27.21806, 56.37778),
    "Dubai Airport (DXB), UAE": (25.25278, 55.36444),
    "Sharjah Airport (SHJ), UAE": (25.32917, 55.51611),
}
LANDMARK_RADIUS_KM = 15.0

RUNS = {
    "CRISIS (Mar 19-24, 2026)": "data/raw/hormuz_crisis_mar2026_ais_positions.csv",
    "CONTROL (Jan 17-21, 2026)": "data/raw/hormuz_control_mar2026_ais_positions.csv",
}


def check_positions(ais_df):
    """Return ais_df with on_land, nearest_landmark, landmark_distance_km,
    near_landmark columns added."""
    df = ais_df.copy()
    df["on_land"] = [globe.is_land(lat, lon) for lat, lon in zip(df["lat"], df["lon"])]

    nearest_name = []
    nearest_dist = []
    for lat, lon in zip(df["lat"], df["lon"]):
        best_name, best_dist = None, float("inf")
        for name, (llat, llon) in LANDMARKS.items():
            d = haversine_km(lat, lon, llat, llon)
            if d < best_dist:
                best_name, best_dist = name, d
        nearest_name.append(best_name)
        nearest_dist.append(best_dist)

    df["nearest_landmark"] = nearest_name
    df["landmark_distance_km"] = nearest_dist
    df["near_landmark"] = df["landmark_distance_km"] <= LANDMARK_RADIUS_KM
    return df


if __name__ == "__main__":
    for label, path in RUNS.items():
        ais_df = pd.read_csv(path, parse_dates=["timestamp"])
        n = len(ais_df)
        checked = check_positions(ais_df)

        on_land = checked[checked["on_land"]]
        near_lm = checked[checked["near_landmark"]]

        print(f"=== {label} === total AIS hourly-bin positions: {n}")
        print(f"  on-land positions: {len(on_land)} ({100 * len(on_land) / n:.2f}%)")
        print(f"  within {LANDMARK_RADIUS_KM:.0f}km of a tracked landmark: "
              f"{len(near_lm)} ({100 * len(near_lm) / n:.2f}%)")

        if len(on_land):
            print("  --- on-land records ---")
            cols = ["mmsi", "timestamp", "lat", "lon", "ship_name", "flag", "vessel_type"]
            print(on_land[cols].to_string(index=False))

        if len(near_lm):
            print("  --- near-landmark records ---")
            cols = ["mmsi", "timestamp", "lat", "lon", "ship_name",
                    "nearest_landmark", "landmark_distance_km"]
            print(near_lm[cols].sort_values("landmark_distance_km").to_string(index=False))

        print()
