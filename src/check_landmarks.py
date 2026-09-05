"""Standalone landmark-proximity check (does not touch match.py).

The first version of this check (src/check_implausible_positions.py) reused
the Hormuz strait bbox pull (25.5-27.0N, 55.5-57.5E), which does not actually
reach any of the four landmarks of interest -- Bushehr sits ~490km outside
that box, and the three airports sit right at or just past its edges. That
made the "0 flagged" result meaningless (there was no data there to flag).

This version pulls fresh AIS data in a small bbox centered on each landmark
directly, for the same crisis/control date windows, and checks distance from
every returned AIS position to that landmark's exact coordinates.
"""
import pandas as pd

from src.fetch_ais import fetch_ais_positions
from src.match import haversine_km

LANDMARKS = {
    "Bushehr nuclear plant, Iran": (28.9669, 50.8878),
    "Bandar Abbas Airport (BND), Iran": (27.21806, 56.37778),
    "Dubai Airport (DXB), UAE": (25.25278, 55.36444),
    "Sharjah Airport (SHJ), UAE": (25.32917, 55.51611),
}
RADIUS_KM = 15.0
# Tighter tiers, computed from the same pull: all four of these landmarks sit
# near real port cities/coastal shipping lanes, so a flat 15km radius mostly
# just measures ordinary background port traffic density, not "positioned at
# the airport/plant." A tight tier (on the facility itself) is the one that
# actually speaks to the Scientific American "ships at airports" pattern.
TIGHT_TIERS_KM = [2.0, 5.0, 15.0]
# +/-0.25 deg is ~28km at these latitudes -- comfortably covers a 15km-radius
# circle around the center point in both lat and lon.
HALF_WIDTH_DEG = 0.25

WINDOWS = {
    "CRISIS (Mar 19-24, 2026)": ("2026-03-19", "2026-03-24"),
    "CONTROL (Jan 17-21, 2026)": ("2026-01-17", "2026-01-21"),
}


def landmark_bbox(lat, lon, half_width=HALF_WIDTH_DEG):
    return {
        "min_lat": lat - half_width,
        "max_lat": lat + half_width,
        "min_lon": lon - half_width,
        "max_lon": lon + half_width,
    }


if __name__ == "__main__":
    tight_rows = []  # only the <=2km tier gets a full record dump -- that's
                      # small and actually landmark-specific; 15km is not.
    window_tier_totals = {}  # (window_label, tier_km) -> (flagged, total)

    for window_label, (start, end) in WINDOWS.items():
        print(f"=== {window_label} ===")
        tier_flagged = {t: 0 for t in TIGHT_TIERS_KM}
        window_total_rows = 0

        for landmark_name, (llat, llon) in LANDMARKS.items():
            bbox = landmark_bbox(llat, llon)
            ais_df = fetch_ais_positions(bbox, start, end)
            n = len(ais_df)
            window_total_rows += n

            if n == 0:
                print(f"  {landmark_name}: 0 AIS rows returned in {HALF_WIDTH_DEG*2:.2f}deg "
                      f"box around ({llat},{llon}) -- no traffic pulled here at all")
                continue

            dist = haversine_km(ais_df["lat"].values, ais_df["lon"].values, llat, llon)
            ais_df = ais_df.assign(landmark_distance_km=dist)

            counts = {t: int((ais_df["landmark_distance_km"] <= t).sum()) for t in TIGHT_TIERS_KM}
            for t in TIGHT_TIERS_KM:
                tier_flagged[t] += counts[t]
            tiers_str = ", ".join(f"<={t:.0f}km: {counts[t]} ({100*counts[t]/n:.2f}%)" for t in TIGHT_TIERS_KM)
            print(f"  {landmark_name}: {n} AIS rows pulled -- {tiers_str}")

            tight = ais_df[ais_df["landmark_distance_km"] <= TIGHT_TIERS_KM[0]]
            if len(tight):
                cols = ["mmsi", "timestamp", "lat", "lon", "ship_name",
                        "flag", "vessel_type", "landmark_distance_km"]
                tight = tight.assign(landmark=landmark_name, window=window_label)
                tight_rows.append(tight)

        for t in TIGHT_TIERS_KM:
            window_tier_totals[(window_label, t)] = (tier_flagged[t], window_total_rows)
            print(f"  --- {window_label} total <= {t:.0f}km: {tier_flagged[t]} / {window_total_rows} "
                  f"({100 * tier_flagged[t] / window_total_rows:.2f}%) ---")
        print()

    print(f"=== SIDE-BY-SIDE (all 4 landmarks combined) ===")
    print(f"{'tier':<10}{'crisis':>22}{'control':>22}")
    for t in TIGHT_TIERS_KM:
        c_flag, c_tot = window_tier_totals[("CRISIS (Mar 19-24, 2026)", t)]
        k_flag, k_tot = window_tier_totals[("CONTROL (Jan 17-21, 2026)", t)]
        print(f"<={t:>5.0f}km  {c_flag:>8}/{c_tot:<6} ({100*c_flag/c_tot:5.2f}%)"
              f"   {k_flag:>8}/{k_tot:<6} ({100*k_flag/k_tot:5.2f}%)")
    print()

    if tight_rows:
        combined = pd.concat(tight_rows, ignore_index=True)
        print(f"=== FULL RECORD LIST, <= {TIGHT_TIERS_KM[0]:.0f}km tier only (both windows) ===")
        cols = ["window", "landmark", "mmsi", "timestamp", "lat", "lon",
                "ship_name", "landmark_distance_km"]
        print(combined[cols].sort_values(["window", "landmark_distance_km"]).to_string(index=False))
    else:
        print(f"No AIS positions fell within {TIGHT_TIERS_KM[0]:.0f}km of any tracked landmark in either window.")
