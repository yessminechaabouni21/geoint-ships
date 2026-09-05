"""Read-only inspection of the 14 crisis-window near-port trajectory-deviation
events (data/processed/hormuz_anchorage_confound_check.csv, window=CRISIS,
near_port=True) to test whether each looks like an STS-transfer rendezvous
(approach -> stop/idle near another vessel -> resume) or ordinary solo
anchorage entry (steady approach, no nearby vessel, no idle stop).

Does not modify hormuz_anchorage_confound_check.csv, hormuz_trajectory_
deviation_v2.csv, trajectory_predict.py, or match.py -- reads them only.
"""
import folium
import numpy as np
import pandas as pd

from src.match import haversine_km

CONFOUND_CHECK_PATH = "data/processed/hormuz_anchorage_confound_check.csv"
AIS_PATH = "data/raw/hormuz_crisis_mar2026_ais_positions.csv"

WINDOW_HOURS = 12
RENDEZVOUS_RADIUS_KM = 2.0
RENDEZVOUS_TIME_WINDOW_HOURS = 1.0
IDLE_SPEED_KMH = 2.0  # roughly walking pace -- below this, call it "stopped/idling"

OUT_MAP_PATH = "data/processed/hormuz_near_port_event_inspection.html"

EVENT_COLORS = [
    "red", "blue", "green", "purple", "orange", "darkred", "cadetblue",
    "darkgreen", "darkblue", "darkpurple", "pink", "gray", "black", "lightred",
]


def load_events():
    df = pd.read_csv(CONFOUND_CHECK_PATH, parse_dates=["predicted_timestamp"])
    events = df[(df["window"].str.contains("CRISIS")) & (df["near_port"] == True)].copy()  # noqa: E712
    events = events.reset_index(drop=True)
    events["event_id"] = events.index
    return events


def vessel_window_track(ais_df, mmsi, center_ts):
    lo = center_ts - pd.Timedelta(hours=WINDOW_HOURS)
    hi = center_ts + pd.Timedelta(hours=WINDOW_HOURS)
    track = ais_df[(ais_df["mmsi"] == mmsi) & (ais_df["timestamp"] >= lo) & (ais_df["timestamp"] <= hi)]
    return track.sort_values("timestamp").reset_index(drop=True)


def track_speeds_kmh(track):
    """Implied speed (km/h) between each consecutive pair in the track.
    Returns array aligned to track rows 1..len-1 (speed INTO that row)."""
    speeds = []
    for i in range(1, len(track)):
        p0, p1 = track.iloc[i - 1], track.iloc[i]
        dt_h = (p1["timestamp"] - p0["timestamp"]).total_seconds() / 3600.0
        if dt_h <= 0:
            speeds.append(np.nan)
            continue
        d = haversine_km(p0["lat"], p0["lon"], p1["lat"], p1["lon"])
        speeds.append(d / dt_h)
    return speeds


def find_rendezvous(ais_df, mmsi, ts, lat, lon):
    """Any OTHER vessel within RENDEZVOUS_RADIUS_KM of (lat,lon), reporting
    within +/- RENDEZVOUS_TIME_WINDOW_HOURS of ts. Returns the closest such
    candidate row (with distance_km, time_gap_hours) or None."""
    lo = ts - pd.Timedelta(hours=RENDEZVOUS_TIME_WINDOW_HOURS)
    hi = ts + pd.Timedelta(hours=RENDEZVOUS_TIME_WINDOW_HOURS)
    window = ais_df[(ais_df["mmsi"] != mmsi) & (ais_df["timestamp"] >= lo) & (ais_df["timestamp"] <= hi)]
    if window.empty:
        return None

    d = haversine_km(lat, lon, window["lat"].values, window["lon"].values)
    window = window.assign(distance_km=d)
    candidates = window[window["distance_km"] <= RENDEZVOUS_RADIUS_KM]
    if candidates.empty:
        return None

    best = candidates.sort_values("distance_km").iloc[0]
    return best


def characterize_speed(track, flagged_ts):
    """Does implied speed drop toward ~0 near the flagged timestamp, or stay
    roughly constant? Compares min speed within +/-2h of the flagged point
    against the track's overall median speed."""
    speeds = track_speeds_kmh(track)
    if len(speeds) < 2:
        return "insufficient data", None, None

    speed_series = pd.Series(speeds, index=track["timestamp"].iloc[1:])
    near_mask = (speed_series.index >= flagged_ts - pd.Timedelta(hours=2)) & \
                (speed_series.index <= flagged_ts + pd.Timedelta(hours=2))
    near_speeds = speed_series[near_mask]
    if near_speeds.empty:
        return "insufficient data", None, None

    min_near = near_speeds.min()
    median_all = speed_series.median()

    if min_near <= IDLE_SPEED_KMH:
        return "speed drops near-zero (stop/idle)", min_near, median_all
    return "speed roughly constant (no stop)", min_near, median_all


if __name__ == "__main__":
    events = load_events()
    ais_df = pd.read_csv(AIS_PATH, parse_dates=["timestamp"])

    m = folium.Map(location=[26.5, 56.2], zoom_start=8, tiles="OpenStreetMap")
    summary_rows = []

    for _, ev in events.iterrows():
        eid = ev["event_id"]
        mmsi = ev["mmsi"]
        ts = ev["predicted_timestamp"]
        color = EVENT_COLORS[eid % len(EVENT_COLORS)]
        label = f"#{eid} {ev['ship_name'] if pd.notna(ev['ship_name']) else mmsi} ({ev['tier_sampled']})"

        track = vessel_window_track(ais_df, mmsi, ts)
        rendezvous = find_rendezvous(ais_df, mmsi, ts, ev["actual_lat"], ev["actual_lon"])
        speed_char, min_near_speed, median_speed = characterize_speed(track, ts)

        layer = folium.FeatureGroup(name=label, show=(eid == 0))

        if len(track) >= 2:
            folium.PolyLine(
                locations=track[["lat", "lon"]].values.tolist(),
                color=color, weight=3, opacity=0.7,
            ).add_to(layer)
            for _, p in track.iterrows():
                folium.CircleMarker(
                    location=[p["lat"], p["lon"]], radius=3, color=color, fill=True, fill_opacity=0.6,
                    popup=f"{label}<br>{p['timestamp']}",
                ).add_to(layer)

        folium.Marker(
            location=[ev["actual_lat"], ev["actual_lon"]],
            icon=folium.Icon(color="red" if ev["tier_sampled"] == "high" else "orange", icon="exclamation-sign"),
            popup=(f"FLAGGED EVENT #{eid}<br>{label}<br>{ts}<br>"
                   f"deviation: {ev['deviation_km']:.1f}km, gap: {ev['gap12_hours']:.0f}h<br>"
                   f"port_distance: {ev['port_distance_km']:.1f}km<br>"
                   f"speed: {speed_char}"),
        ).add_to(layer)

        if rendezvous is not None:
            folium.Marker(
                location=[rendezvous["lat"], rendezvous["lon"]],
                icon=folium.Icon(color="black", icon="screenshot"),
                popup=(f"Rendezvous candidate for event #{eid}<br>"
                       f"MMSI {rendezvous['mmsi']} ({rendezvous.get('ship_name', '')})<br>"
                       f"{rendezvous['timestamp']}, {rendezvous['distance_km']:.2f}km away"),
            ).add_to(layer)

        layer.add_to(m)

        summary_rows.append({
            "event_id": eid,
            "mmsi": mmsi,
            "ship_name": ev["ship_name"],
            "tier": ev["tier_sampled"],
            "flagged_timestamp": ts,
            "deviation_km": round(ev["deviation_km"], 1),
            "gap12_hours": ev["gap12_hours"],
            "port_distance_km": round(ev["port_distance_km"], 1),
            "track_points_in_window": len(track),
            "rendezvous_found": rendezvous is not None,
            "rendezvous_mmsi": rendezvous["mmsi"] if rendezvous is not None else None,
            "rendezvous_distance_km": round(rendezvous["distance_km"], 2) if rendezvous is not None else None,
            "speed_characterization": speed_char,
            "min_speed_near_event_kmh": round(min_near_speed, 1) if min_near_speed is not None else None,
            "median_speed_track_kmh": round(median_speed, 1) if median_speed is not None else None,
        })

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(OUT_MAP_PATH)

    summary = pd.DataFrame(summary_rows)

    print("=== Per-event summary (14 crisis near-port notable/high deviation events) ===")
    print(summary[["event_id", "mmsi", "ship_name", "tier", "flagged_timestamp", "deviation_km",
                    "gap12_hours", "port_distance_km", "rendezvous_found", "rendezvous_distance_km",
                    "speed_characterization"]].to_string(index=False))

    both = summary[summary["rendezvous_found"] & summary["speed_characterization"].str.contains("near-zero")]
    neither = summary[~summary["rendezvous_found"] & ~summary["speed_characterization"].str.contains("near-zero")]

    print(f"\nBoth rendezvous candidate AND speed-drop-to-near-zero "
          f"(strongest combined STS-transfer signature): {len(both)} / {len(summary)}")
    if len(both):
        print(both[["event_id", "mmsi", "ship_name", "tier"]].to_string(index=False))

    print(f"\nNeither rendezvous candidate NOR speed drop "
          f"(more consistent with ordinary anchorage entry): {len(neither)} / {len(summary)}")
    if len(neither):
        print(neither[["event_id", "mmsi", "ship_name", "tier"]].to_string(index=False))

    print(f"\nSaved map to {OUT_MAP_PATH}")
