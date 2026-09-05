"""Diagnostic (does not modify trajectory_predict.py, match.py, or the
existing v2 output): tests whether the crisis-vs-control gap in
trajectory-deviation "notable"/"high" rates is a real behavioral difference
or a confound -- the control window's higher rate could just be ordinary
port-approach/anchorage maneuvering (slowing, turning near a berth) that a
straight-line constant-velocity model naturally flags as "deviation" even
though it's routine, if the crisis-reduced traffic is disproportionately
straight-line strait transit with less near-port movement in the sample.

Reads the already-saved data/processed/hormuz_trajectory_deviation_v2.csv
and data/raw/hormuz_{crisis,control}_mar2026_ais_positions.csv (for vessel
type). Writes its own diagnostic output; does not touch either input.
"""
import pandas as pd

from src.match import haversine_km

# Known anchorages/ports within or immediately adjacent to the Hormuz bbox
# (25.5-27.0N, 55.5-57.5E). Coordinates from Wikipedia infoboxes (Khor
# Fakkan, Khasab, Larak Island) plus the Fujairah/Khor Fakkan STS anchorage
# already used in trajectory_predict.py's hotspot list, plus Qeshm Island
# and Bandar Abbas (the latter sits just north of the bbox edge but is close
# enough that nearby vessels should still be attributed to it).
ANCHORAGES = {
    "Khor Fakkan port, UAE": (25.33917, 56.35611),
    "Fujairah/Khor Fakkan STS anchorage": (25.1500, 56.4500),
    "Khasab port, Oman": (26.20333, 56.24944),
    "Larak Island anchorage, Iran": (26.85333, 56.35556),
    "Qeshm Island port, Iran": (26.9581, 56.2719),
    "Bandar Abbas port, Iran": (27.1365, 56.2808),
}
PORT_RADIUS_KM = 20.0

TRAJ_PATH = "data/processed/hormuz_trajectory_deviation_v2.csv"
AIS_PATHS = {
    "CRISIS (Mar 19-24, 2026)": "data/raw/hormuz_crisis_mar2026_ais_positions.csv",
    "CONTROL (Jan 17-21, 2026)": "data/raw/hormuz_control_mar2026_ais_positions.csv",
}
OUT_PATH = "data/processed/hormuz_anchorage_confound_check.csv"

SAMPLE_N = 50
RANDOM_STATE = 42


def nearest_anchorage_km(lat, lon):
    return min(haversine_km(lat, lon, alat, alon) for alat, alon in ANCHORAGES.values())


if __name__ == "__main__":
    traj = pd.read_csv(TRAJ_PATH, parse_dates=["predicted_timestamp"])

    vessel_type_frames = []
    for label, path in AIS_PATHS.items():
        ais_df = pd.read_csv(path, parse_dates=["timestamp"])
        vt = ais_df[["mmsi", "timestamp", "vessel_type"]].rename(columns={"timestamp": "predicted_timestamp"})
        vt = vt.assign(window=label)
        vessel_type_frames.append(vt)
    vessel_types = pd.concat(vessel_type_frames, ignore_index=True)

    traj = traj.merge(vessel_types, on=["mmsi", "predicted_timestamp", "window"], how="left")

    sampled_all = []
    port_summary = {}
    type_summary = {}

    for window in traj["window"].unique():
        w = traj[traj["window"] == window]
        print(f"=== {window} ===")

        window_rows = []
        for tier in ["notable", "high"]:
            pool = w[w["tier"] == tier]
            if len(pool) > SAMPLE_N:
                sample = pool.sample(n=SAMPLE_N, random_state=RANDOM_STATE)
            else:
                sample = pool
            sample = sample.copy()
            sample["port_distance_km"] = [
                nearest_anchorage_km(lat, lon) for lat, lon in zip(sample["actual_lat"], sample["actual_lon"])
            ]
            sample["near_port"] = sample["port_distance_km"] <= PORT_RADIUS_KM
            sample["tier_sampled"] = tier

            n = len(sample)
            n_near = int(sample["near_port"].sum())
            pct = 100 * n_near / n if n else 0.0
            print(f"  {tier}: sampled {n} of {len(pool)} total, {n_near} ({pct:.1f}%) within "
                  f"{PORT_RADIUS_KM:.0f}km of a known anchorage/port")
            port_summary[(window, tier)] = (n_near, n)

            window_rows.append(sample)

        window_sample = pd.concat(window_rows, ignore_index=True)
        sampled_all.append(window_sample)

        print(f"\n  vessel type breakdown (sampled notable+high, near-port vs open-water):")
        if window_sample["vessel_type"].notna().any():
            vt_table = (
                window_sample.assign(vessel_type=window_sample["vessel_type"].fillna("UNKNOWN"))
                .groupby(["vessel_type", "near_port"])
                .size()
                .unstack(fill_value=0)
            )
            for col in [True, False]:
                if col not in vt_table.columns:
                    vt_table[col] = 0
            vt_table = vt_table.rename(columns={True: "near_port", False: "open_water"})
            vt_table["total"] = vt_table["near_port"] + vt_table["open_water"]
            vt_table["pct_near_port"] = (100 * vt_table["near_port"] / vt_table["total"]).round(1)
            print(vt_table.sort_values("total", ascending=False).to_string())
            type_summary[window] = vt_table
        else:
            print("  no vessel_type data available for this sample")
        print()

    print("=== SIDE-BY-SIDE: % of sampled notable+high events near a known anchorage/port ===")
    print(f"{'window':<28}{'tier':<10}{'near-port %':>14}{'  (n near / n sampled)'}")
    for window in traj["window"].unique():
        for tier in ["notable", "high"]:
            n_near, n = port_summary[(window, tier)]
            pct = 100 * n_near / n if n else 0.0
            print(f"{window:<28}{tier:<10}{pct:>13.1f}%   ({n_near}/{n})")

    combined = pd.concat(sampled_all, ignore_index=True)
    combined.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(combined)} sampled+annotated rows to {OUT_PATH}")
