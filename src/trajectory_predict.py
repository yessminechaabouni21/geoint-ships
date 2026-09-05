"""Trajectory-deviation detection: a standalone, physics-based signal
independent of the SAR cross-check in src/match.py.

For each vessel's AIS track, use a constant-velocity (great-circle bearing +
speed) model from its two most recent hourly-bin positions to predict where
it should be at the next hourly bin, then compare against where it actually
reported. Large deviations are consistent with an undisclosed detour, an
STS-transfer stop, or a spoofed position jump -- this does NOT require a SAR
detection at all, unlike match.py's discrepancy check, so it's a genuinely
independent signal to combine with the SAR-based one later.

Explicitly out of scope per requirements: no deep learning, no Kalman filter
state estimation, no quantum ML. AIS is hourly grid-cell presence bins (see
src/config.py caveat), which doesn't support anything finer than a simple
constant-velocity extrapolation -- a Kalman filter's main advantage (fusing
noisy repeated observations into a smoothed state) buys little when there is
only one observation per hour to begin with.
"""
import math

import numpy as np
import pandas as pd

from src.match import haversine_km, EARTH_RADIUS_KM

# --- Deviation tiers ---------------------------------------------------
# NOT reused from match.py's matched/discrepant/likely_spoofed (1.5/20km).
# Those thresholds were calibrated for "how far is a claimed AIS position
# from an independently-observed SAR position" -- a single fixed-time
# comparison. This check instead asks "how far did a vessel end up from
# where its OWN prior heading/speed says it should be after some elapsed
# gap," which has a wider natural spread: ordinary course/speed changes (a
# turn, a slow-down approaching anchorage, a stop) can offset a naive
# constant-velocity projection by several km even over a single hour with
# zero spoofing, especially for faster vessels.
#
# Gap-scaled thresholds: a fixed km cutoff doesn't work once gaps vary --
# a vessel reappearing 100km off course after a 1-hour gap is a strong
# anomaly, but 100km off after a 20-hour gap (an AIS-silent stretch, common
# during the crisis -- see the unmatched/blockade findings) is well within
# ordinary route flexibility for even a slow tanker. The tolerance needs to
# grow with gap length, but NOT linearly with a vessel's own top speed --
# that would treat "the vessel could theoretically have sailed anywhere
# within reach" as "normal," which defeats the point of the check. Instead
# this uses sqrt(gap_hours) scaling: a random-walk/diffusive-style growth
# rate, calibrated so it reproduces the original fixed 5km/25km cutoffs
# exactly at gap_hours=1, and grows more slowly than linear thereafter (a
# 16-hour gap gets 4x the 1-hour tolerance, not 16x) -- generous enough to
# stop penalizing ordinary long AIS-silent gaps, but still discriminating at
# gaps of a day or more rather than the threshold ballooning to uselessness.
NORMAL_RATE_KM_PER_SQRT_HR = 5.0
NOTABLE_RATE_KM_PER_SQRT_HR = 25.0


def _tier_thresholds(gap_hours):
    scale = math.sqrt(gap_hours)
    return NORMAL_RATE_KM_PER_SQRT_HR * scale, NOTABLE_RATE_KM_PER_SQRT_HR * scale

HOTSPOT_RADIUS_KM = 50.0

# Approximate named-water anchor points for known STS-transfer hotspots.
# These are standard geographic coordinates for the named places, not
# incident-specific claims -- Gulf of Oman is anchored on the Fujairah/Khor
# Fakkan anchorage, which is independently documented as an active STS zone
# in 2026 reporting (DeepDraft, Misbar). The other four are outside the
# Hormuz bbox this module is first tested on, so they'll trivially return
# zero hits here -- included for when the pipeline covers other regions.
STS_HOTSPOTS = {
    "Ceuta / Strait of Gibraltar": (35.8894, -5.3213),
    "Kalamata / Messenian Gulf": (36.9000, 22.1000),
    "Laconian Gulf": (36.5000, 22.8500),
    "Gulf of Oman (Fujairah/Khor Fakkan anchorage)": (25.1500, 56.4500),
    "Laoag / Luzon Strait": (18.3000, 120.5000),
}


def _bearing_deg(lat1, lon1, lat2, lon2):
    lat1r, lon1r, lat2r, lon2r = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2r - lon1r
    x = math.sin(dlon) * math.cos(lat2r)
    y = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _destination_point(lat1, lon1, bearing_deg, distance_km):
    R = EARTH_RADIUS_KM
    lat1r, lon1r, brng = math.radians(lat1), math.radians(lon1), math.radians(bearing_deg)
    ang = distance_km / R
    lat2r = math.asin(math.sin(lat1r) * math.cos(ang) + math.cos(lat1r) * math.sin(ang) * math.cos(brng))
    lon2r = lon1r + math.atan2(
        math.sin(brng) * math.sin(ang) * math.cos(lat1r),
        math.cos(ang) - math.sin(lat1r) * math.sin(lat2r),
    )
    return math.degrees(lat2r), math.degrees(lon2r)


def _classify(distance_km, gap_hours):
    normal_thr, notable_thr = _tier_thresholds(gap_hours)
    if distance_km <= normal_thr:
        return "normal"
    if distance_km <= notable_thr:
        return "notable"
    return "high"


def _nearest_hotspot(lat, lon):
    best_name, best_dist = None, float("inf")
    for name, (hlat, hlon) in STS_HOTSPOTS.items():
        d = haversine_km(lat, lon, hlat, hlon)
        if d < best_dist:
            best_name, best_dist = name, d
    return best_name, best_dist


def predict_deviations(ais_df):
    """For each vessel, walk consecutive AIS positions (however many hours
    apart -- gaps are not skipped or required to be exactly 1h) and, for
    each triple (p0, p1, p2), use p0->p1 to get a constant-velocity
    heading/speed, then project forward across the ACTUAL elapsed time to
    p2 and compare against where the vessel actually reappeared.

    This intentionally does NOT require 1-hour-spaced reporting: a vessel
    that goes AIS-silent for hours (common during the crisis window -- see
    the unmatched/blockade findings) still gets evaluated on reappearance,
    just against a gap-scaled tolerance (see _tier_thresholds) rather than
    being silently dropped from the sample.

    Returns one row per (mmsi, predicted timestamp) triple evaluated, with
    gap01_hours (speed/heading source interval) and gap12_hours (prediction
    horizon actually extrapolated across) both recorded.
    """
    df = ais_df.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)
    rows = []

    for mmsi, track in df.groupby("mmsi"):
        track = track.reset_index(drop=True)
        if len(track) < 3:
            continue

        for i in range(2, len(track)):
            p0, p1, p2 = track.iloc[i - 2], track.iloc[i - 1], track.iloc[i]

            gap01_hours = (p1["timestamp"] - p0["timestamp"]).total_seconds() / 3600.0
            gap12_hours = (p2["timestamp"] - p1["timestamp"]).total_seconds() / 3600.0
            if gap01_hours <= 0 or gap12_hours <= 0:
                continue  # duplicate/out-of-order timestamps -- speed undefined

            dist01 = haversine_km(p0["lat"], p0["lon"], p1["lat"], p1["lon"])
            speed_kmh = 0.0
            if dist01 == 0:
                pred_lat, pred_lon = p1["lat"], p1["lon"]
            else:
                bearing = _bearing_deg(p0["lat"], p0["lon"], p1["lat"], p1["lon"])
                speed_kmh = dist01 / gap01_hours
                pred_lat, pred_lon = _destination_point(p1["lat"], p1["lon"], bearing, speed_kmh * gap12_hours)

            deviation_km = haversine_km(pred_lat, pred_lon, p2["lat"], p2["lon"])
            tier = _classify(deviation_km, gap12_hours)

            near_name, near_dist = (None, None)
            if tier == "high":
                near_name, near_dist = _nearest_hotspot(p2["lat"], p2["lon"])

            rows.append({
                "mmsi": mmsi,
                "ship_name": p2.get("ship_name"),
                "predicted_timestamp": p2["timestamp"],
                "predicted_lat": pred_lat, "predicted_lon": pred_lon,
                "actual_lat": p2["lat"], "actual_lon": p2["lon"],
                "implied_speed_kmh": speed_kmh,
                "gap01_hours": gap01_hours,
                "gap12_hours": gap12_hours,
                "deviation_km": deviation_km,
                "tier": tier,
                "near_hotspot": near_name if near_dist is not None and near_dist <= HOTSPOT_RADIUS_KM else None,
                "hotspot_distance_km": near_dist if near_dist is not None and near_dist <= HOTSPOT_RADIUS_KM else None,
            })

    return pd.DataFrame(rows)


RUNS = {
    "CRISIS (Mar 19-24, 2026)": "hormuz_crisis_mar2026",
    "CONTROL (Jan 17-21, 2026)": "hormuz_control_mar2026",
}

TIER_ORDER = ["normal", "notable", "high"]

if __name__ == "__main__":
    all_results = []
    summaries = {}

    for label, run_label in RUNS.items():
        ais_df = pd.read_csv(f"data/raw/{run_label}_ais_positions.csv", parse_dates=["timestamp"])
        result = predict_deviations(ais_df)
        result = result.assign(window=label)
        all_results.append(result)

        n = len(result)
        vc = result["tier"].value_counts()
        summaries[label] = (n, vc)

        # How many vessels had >=1 gap of more than 1 hour between the pair
        # used for speed/heading and the point being predicted -- these are
        # exactly the vessels the old strict-consecutive check dropped
        # entirely (it required gap12 == exactly 3600s for every triple).
        gappy = result[result["gap12_hours"] > 1.0]
        n_gappy_vessels = gappy["mmsi"].nunique()
        n_all_vessels = result["mmsi"].nunique()

        print(f"=== {label} === vessels evaluated: {n_all_vessels}, "
              f"predicted points: {n}")
        print(f"  vessels with >=1 prediction across a gap >1h "
              f"(previously excluded entirely by the strict-consecutive check): "
              f"{n_gappy_vessels}")
        print(f"  median gap12_hours: {result['gap12_hours'].median():.2f}, "
              f"max: {result['gap12_hours'].max():.2f}")
        for t in TIER_ORDER:
            c = vc.get(t, 0)
            print(f"  {t}: {c} ({100 * c / n:.1f}%)" if n else f"  {t}: 0")

        high = result[result["tier"] == "high"]
        high_after_gap = high[high["gap12_hours"] > 1.0]
        print(f"  high-tier events that occurred after a gap >1h "
              f"(would have been invisible under the old strict check): "
              f"{len(high_after_gap)} / {len(high)}")

        near_hotspot = high[high["near_hotspot"].notna()]
        print(f"  high-deviation events near a known STS hotspot (<={HOTSPOT_RADIUS_KM:.0f}km): "
              f"{len(near_hotspot)} / {len(high)} high-tier events")
        if len(near_hotspot):
            cols = ["mmsi", "ship_name", "predicted_timestamp", "gap12_hours", "actual_lat", "actual_lon",
                    "deviation_km", "near_hotspot", "hotspot_distance_km"]
            print(near_hotspot[cols].sort_values("hotspot_distance_km").to_string(index=False))
        print()

    print("=== SIDE-BY-SIDE ===")
    header = f"{'tier':<10}" + "".join(f"{label:>28}" for label in summaries)
    print(header)
    for t in TIER_ORDER:
        row = f"{t:<10}"
        for label, (n, vc) in summaries.items():
            c = vc.get(t, 0)
            pct = 100 * c / n if n else 0.0
            row += f"{c:>10} ({pct:5.1f}%)".rjust(28)
        print(row)

    combined = pd.concat(all_results, ignore_index=True)
    out_path = "data/processed/hormuz_trajectory_deviation_v2.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nSaved {len(combined)} rows to {out_path}")
