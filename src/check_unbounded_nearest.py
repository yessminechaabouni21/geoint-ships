"""Standalone diagnostic (does not touch match.py): for SAR detections the
existing pipeline classified "unmatched" (no MMSI at all -- GFW's own SAR/AIS
correlation found no candidate), search ALL AIS activity in the same hourly
bin, across every vessel, with no distance cap and no requirement that the
MMSI already be attached to that SAR detection.

This tests directly whether "unmatched" is hiding a far-away spoofed AIS
report (some vessel WAS broadcasting nearby-ish, just too far for GFW's own
correlation to link it) versus true silence (no AIS activity anywhere in the
bbox during that hour at all).
"""
import numpy as np
import pandas as pd

from src.match import haversine_km

SAR_CLASSIFIED_PATH = "data/processed/hormuz_crisis_mar2026_classified.csv"
AIS_RAW_PATH = "data/raw/hormuz_crisis_mar2026_ais_positions.csv"


def nearest_unbounded(sar_df, ais_df):
    """For each SAR row, find the nearest AIS position among ALL vessels in
    the same hourly bin (sar_timestamp floored to the hour == ais timestamp).

    Returns sar_df with columns: hour_bin, ais_candidates_in_bin,
    nearest_mmsi, nearest_distance_km added.
    """
    ais_df = ais_df.copy()
    ais_df["hour_bin"] = ais_df["timestamp"].dt.floor("h")

    sar_df = sar_df.copy()
    sar_df["hour_bin"] = sar_df["sar_timestamp"].dt.floor("h")

    n_candidates = []
    nearest_mmsi = []
    nearest_dist = []

    for _, row in sar_df.iterrows():
        bin_ais = ais_df[ais_df["hour_bin"] == row["hour_bin"]]
        n_candidates.append(len(bin_ais))
        if bin_ais.empty:
            nearest_mmsi.append(None)
            nearest_dist.append(None)
            continue
        d = haversine_km(row["sar_lat"], row["sar_lon"],
                          bin_ais["lat"].values, bin_ais["lon"].values)
        idx = np.argmin(d)
        nearest_mmsi.append(bin_ais.iloc[idx]["mmsi"])
        nearest_dist.append(float(d[idx]))

    sar_df["ais_candidates_in_bin"] = n_candidates
    sar_df["nearest_mmsi"] = nearest_mmsi
    sar_df["nearest_distance_km"] = nearest_dist
    return sar_df


if __name__ == "__main__":
    sar_df = pd.read_csv(SAR_CLASSIFIED_PATH, parse_dates=["sar_timestamp"])
    ais_df = pd.read_csv(AIS_RAW_PATH, parse_dates=["timestamp"])

    unmatched = sar_df[sar_df["classification"] == "unmatched"].copy()
    print(f"Unmatched SAR detections in crisis window: {len(unmatched)}")

    result = nearest_unbounded(unmatched, ais_df)

    no_ais_in_bin = result[result["ais_candidates_in_bin"] == 0]
    has_ais_in_bin = result[result["ais_candidates_in_bin"] > 0]

    print(f"  hour-bins with ZERO AIS activity anywhere in bbox (true silence): "
          f"{len(no_ais_in_bin)} ({100 * len(no_ais_in_bin) / len(result):.1f}%)")
    print(f"  hour-bins with >=1 AIS position somewhere in bbox (nearest match found): "
          f"{len(has_ais_in_bin)} ({100 * len(has_ais_in_bin) / len(result):.1f}%)")

    if len(has_ais_in_bin):
        d = has_ais_in_bin["nearest_distance_km"].dropna().sort_values().values
        pct = np.percentile(d, [25, 50, 75, 90, 95, 100])
        print(f"\n  Unbounded nearest-distance distribution (km), n={len(d)}:")
        print(f"  25th/50th/75th/90th/95th/max: {np.round(pct, 2)}")
        for thr in [4.5, 20, 50, 100]:
            c = (d > thr).sum()
            print(f"  >{thr}km: {c} ({100 * c / len(d):.1f}%)")

        print("\n  Full record list (previously 'unmatched', now with nearest AIS candidate):")
        cols = ["sar_id", "sar_timestamp", "sar_lat", "sar_lon",
                "ais_candidates_in_bin", "nearest_mmsi", "nearest_distance_km"]
        print(has_ais_in_bin[cols].sort_values("nearest_distance_km").to_string(index=False))

    if len(no_ais_in_bin):
        print("\n  Records with zero AIS activity in bbox during that hour (true silence):")
        cols = ["sar_id", "sar_timestamp", "sar_lat", "sar_lon", "ais_candidates_in_bin"]
        print(no_ais_in_bin[cols].to_string(index=False))
