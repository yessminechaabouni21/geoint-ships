"""Diagnostic checks on the classified output: speed sanity, distance
distribution vs. threshold, and date/location clustering -- to distinguish a
real signal from residual hourly-bin-to-SAR-timestamp noise.
"""
import inspect

import numpy as np
import pandas as pd

from src.match import match_and_classify, haversine_km

KNOTS_PER_KMH = 1 / 1.852
RAS_LAFFAN = (25.9, 51.5)
RAS_LAFFAN_RADIUS_KM = 20
TANKER_CARGO_TYPES = {"CARGO", "TANKER", "OTHER_NON_FISHING"}  # GFW lumps many non-fishing merchant vessels here


def speed_sanity_check(match_df, top_n=10):
    print(f"\n{'='*70}\n1. SPEED SANITY CHECK (top {top_n} largest-distance discrepant cases)\n{'='*70}")

    d = match_df[(match_df["classification"] == "discrepant") & match_df["distance_km"].notna()].copy()
    d = d.sort_values("distance_km", ascending=False).head(top_n)

    d["speed_knots"] = (d["distance_km"] / (d["time_gap_seconds"] / 3600)) * KNOTS_PER_KMH
    d["flag_suspicious"] = d["vessel_type"].isin(TANKER_CARGO_TYPES) & (d["speed_knots"] > 20)

    cols = ["mmsi", "ship_name", "vessel_type", "flag", "distance_km", "time_gap_seconds", "speed_knots", "flag_suspicious"]
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(d[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    n_flagged = d["flag_suspicious"].sum()
    if n_flagged:
        print(f"\n[!] {n_flagged} case(s) imply >20kt for a tanker/cargo-type vessel -- possible remaining bug.")
    else:
        print("\nNo cases imply >20kt for a tanker/cargo-type vessel. Speeds are consistent with the ~28min "
              "hourly-bin-to-SAR-timestamp gap, not a matching bug.")
    return d


def distance_distribution(match_df):
    sig = inspect.signature(match_and_classify)
    distance_tolerance_km = sig.parameters["distance_tolerance_km"].default
    time_tolerance_seconds = sig.parameters["time_tolerance_seconds"].default

    print(f"\n{'='*70}\n2. DISTANCE DISTRIBUTION (all cases with an AIS candidate found)\n{'='*70}")
    print(f"Current threshold in match.py: distance_tolerance_km={distance_tolerance_km}, "
          f"time_tolerance_seconds={time_tolerance_seconds}")

    d = match_df[match_df["distance_km"].notna()].copy()
    dist = d["distance_km"]

    percentiles = [25, 50, 75, 90, 95, 100]
    pct_values = np.percentile(dist, percentiles)
    print("\nPercentiles (km):")
    for p, v in zip(percentiles, pct_values):
        label = "max" if p == 100 else f"p{p}"
        print(f"  {label:>4}: {v:6.2f} km")

    print(f"\n  n={len(dist)}  mean={dist.mean():.2f} km  std={dist.std():.2f} km")

    print(f"\nHistogram (bin width 2km, threshold at {distance_tolerance_km}km marked with |):")
    max_bin = int(np.ceil(dist.max() / 2) * 2) + 2
    bins = np.arange(0, max_bin + 2, 2)
    counts, edges = np.histogram(dist, bins=bins)
    max_count = counts.max() if len(counts) else 1
    for i, c in enumerate(counts):
        lo, hi = edges[i], edges[i + 1]
        marker = " <-- threshold" if lo <= distance_tolerance_km < hi else ""
        bar = "#" * int(40 * c / max_count) if max_count else ""
        print(f"  [{lo:5.1f},{hi:5.1f}) {c:4d} {bar}{marker}")

    # Check whether discrepant cases form a distinct cluster or hug the threshold
    disc = d[d["classification"] == "discrepant"]["distance_km"]
    near_threshold = disc[(disc > distance_tolerance_km) & (disc <= distance_tolerance_km * 3)]
    print(f"\nOf {len(disc)} discrepant cases with a distance value, {len(near_threshold)} "
          f"({100*len(near_threshold)/len(disc):.0f}%) fall within {distance_tolerance_km}-"
          f"{distance_tolerance_km*3}km (i.e. just above the cutoff -- consistent with noise).")
    far = disc[disc > distance_tolerance_km * 3]
    print(f"{len(far)} ({100*len(far)/len(disc):.0f}%) exceed {distance_tolerance_km*3}km -- a distinct, "
          f"harder-to-explain-as-noise cluster.")
    return d


def date_location_breakdown(match_df):
    print(f"\n{'='*70}\n3. DATE / LOCATION BREAKDOWN of discrepant cases (Ras Laffan ~{RAS_LAFFAN}, "
          f"Oct 4-6 incident window)\n{'='*70}")

    all_df = match_df.copy()
    all_df["sar_timestamp"] = pd.to_datetime(all_df["sar_timestamp"], utc=True)
    all_df["date"] = all_df["sar_timestamp"].dt.date

    print("\nBy date, ALL SAR detections in the pull (not just discrepant):")
    print(all_df.groupby("date").size().rename("count").to_string())

    unique_dates = all_df["date"].nunique()
    if unique_dates == 1:
        print(f"\nNOTE: every SAR detection in this Oct 3-7 pull comes from a SINGLE satellite pass "
              f"({all_df['sar_timestamp'].iloc[0]}). Sentinel-1 does not image the same spot daily -- "
              f"its revisit interval over this bbox is longer than the 5-day window we queried, so there is "
              f"no multi-day comparison possible here; a day-by-day cluster check is moot with only one pass. "
              f"That single pass does fall inside the documented Oct 4-6 window, which is the most this data "
              f"can say about timing.")

    d = all_df[all_df["classification"] == "discrepant"].copy()
    d["dist_to_ras_laffan_km"] = haversine_km(d["sar_lat"], d["sar_lon"], RAS_LAFFAN[0], RAS_LAFFAN[1])
    d["near_ras_laffan"] = d["dist_to_ras_laffan_km"] <= RAS_LAFFAN_RADIUS_KM

    all_df["dist_to_ras_laffan_km"] = haversine_km(all_df["sar_lat"], all_df["sar_lon"], RAS_LAFFAN[0], RAS_LAFFAN[1])
    all_df["near_ras_laffan"] = all_df["dist_to_ras_laffan_km"] <= RAS_LAFFAN_RADIUS_KM

    print(f"\nSpatial clustering near Ras Laffan (<= {RAS_LAFFAN_RADIUS_KM}km), discrepant vs. baseline:")
    baseline_rate = all_df["near_ras_laffan"].mean()
    matched_rate = match_df.loc[match_df["classification"] == "matched"]
    matched_near = haversine_km(matched_rate["sar_lat"], matched_rate["sar_lon"], RAS_LAFFAN[0], RAS_LAFFAN[1]) <= RAS_LAFFAN_RADIUS_KM
    matched_rate_val = matched_near.mean() if len(matched_rate) else float("nan")
    discrepant_rate = d["near_ras_laffan"].mean()

    print(f"  ALL {len(all_df)} SAR detections:      {all_df['near_ras_laffan'].sum():3d} near Ras Laffan ({100*baseline_rate:.0f}%)")
    print(f"  {len(matched_rate)} matched cases:            {int(matched_near.sum()):3d} near Ras Laffan ({100*matched_rate_val:.0f}%)")
    print(f"  {len(d)} discrepant cases:          {int(d['near_ras_laffan'].sum()):3d} near Ras Laffan ({100*discrepant_rate:.0f}%)")

    if discrepant_rate > baseline_rate * 1.3:
        print(f"\n  Discrepant rate near Ras Laffan ({100*discrepant_rate:.0f}%) is notably higher than the "
              f"overall base rate ({100*baseline_rate:.0f}%) -- some spatial concentration, consistent with "
              f"(but not proof of) a real localized effect.")
    else:
        print(f"\n  Discrepant rate near Ras Laffan ({100*discrepant_rate:.0f}%) is NOT meaningfully higher than "
              f"the overall base rate ({100*baseline_rate:.0f}%) -- discrepant cases are not distinctly "
              f"clustered there relative to where detections occur in general.")

    print(f"\n{len(d) - int(d['near_ras_laffan'].sum())} of {len(d)} discrepant cases are outside the "
          f"{RAS_LAFFAN_RADIUS_KM}km radius, i.e. spread across the rest of the bounding box.")
    return d


if __name__ == "__main__":
    from src.config import TEST_BBOX, TEST_START_DATE, TEST_END_DATE
    from src.fetch_sar import fetch_sar_detections
    from src.fetch_ais import fetch_ais_positions

    sar_df = fetch_sar_detections(TEST_BBOX, TEST_START_DATE, TEST_END_DATE)
    ais_df = fetch_ais_positions(TEST_BBOX, TEST_START_DATE, TEST_END_DATE)
    match_df = match_and_classify(sar_df, ais_df)

    speed_sanity_check(match_df, top_n=10)
    distance_distribution(match_df)
    date_location_breakdown(match_df)
