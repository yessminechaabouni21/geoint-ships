"""Interactive retrospective dashboard for the maritime-anomaly project.

SHIPS ONLY. No planes / subsea cables / flood data (not built or validated).
This is a presentation / exploration layer on top of already-generated,
fully-verified output files -- it makes NO API calls and pulls NO new data.

Scoring source of truth: data/processed/vessel_reliability_scores_v6.csv
(the FINAL, fully-audited leaderboard -- every top-15 vessel externally
verified against real ship registries). v1-v5 are referenced only in the
Methodology panel, for before/after context.

Run:  streamlit run src/dashboard.py
      (a plain `python src/dashboard.py` just prints the data-file audit)
"""
import base64
import glob
import io
import json
import math
import os

import numpy as np
import pandas as pd
import pydeck as pdk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROC = "data/processed"
RAW = "data/raw"
FIGS = "figures"
SCORES_FILE = f"{PROC}/vessel_reliability_scores_v6.csv"   # FINAL

# --------------------------------------------------------------------------
# Window registry -- each maps to an already-saved classified/reclassified
# (or, for India, re-linked) CSV. No window triggers a fetch.
# --------------------------------------------------------------------------
WINDOWS = {
    "Qatar incident (Oct 2025)":
        dict(file="qatar_gnss_spoofing_oct2025_classified.csv", kind="classified"),
    "Qatar control (Sep 2025)":
        dict(file="qatar_control_sept2025_classified.csv", kind="classified"),
    "Hormuz crisis (Mar 2026)":
        dict(file="hormuz_crisis_mar2026_reclassified.csv", kind="classified"),
    "Hormuz control (Jan 2026)":
        dict(file="hormuz_control_mar2026_reclassified.csv", kind="classified"),
    "India baseline / New Mangalore (Sep-Oct 2025)":
        dict(file="india_mangalore_relinked.csv", kind="relinked"),
    "Jamnagar / Vadinar — Nayara refinery (Feb 2026)":
        dict(file="jamnagar_vadinar_feb2026_classified.csv", kind="classified"),
}

CANON = ["matched", "discrepant", "likely_spoofed", "unmatched"]
CLS_COLORS = {
    "matched": "#2a9d8f",       # green
    "discrepant": "#e9c46a",    # yellow
    "likely_spoofed": "#c1121f",  # red
    "unmatched": "#9aa4ab",     # gray
}
# Same palette as [r, g, b] for deck.gl layers (icon tint, halo, trail).
CLS_RGB = {
    "matched": [42, 157, 143],
    "discrepant": [233, 196, 106],
    "likely_spoofed": [193, 18, 31],
    "unmatched": [154, 164, 171],
}
# India's re-linking layer is not a spoofing classification -- map its
# per-detection link outcome onto the same 4 buckets for a common view.
RELINK_TO_CANON = {
    "gfw_locked": "matched",              # GFW's given MMSI verified & kept
    "relinked": "discrepant",             # re-assigned to a closer vessel
    "forced_low_confidence": "likely_spoofed",
    "dark_vessel": "unmatched",           # no AIS-tracked vessel explains it
}

# Each window's already-cached raw AIS position file (used ONLY to draw the
# historical track polyline behind each SAR detection -- no fetch, no new data).
RAW_AIS_LABEL = {
    "Qatar incident (Oct 2025)": "qatar_gnss_spoofing_oct2025",
    "Qatar control (Sep 2025)": "qatar_control_sept2025",
    "Hormuz crisis (Mar 2026)": "hormuz_crisis_mar2026",
    "Hormuz control (Jan 2026)": "hormuz_control_mar2026",
    "India baseline / New Mangalore (Sep-Oct 2025)": "india_mangalore_sep_oct2025",
    "Jamnagar / Vadinar — Nayara refinery (Feb 2026)": "jamnagar_vadinar_feb2026",
}
INDIA_WINDOW = "India baseline / New Mangalore (Sep-Oct 2025)"

# --------------------------------------------------------------------------
# Loiter / dwell detector output (src/loiter_detector.py -> loiter_flags_*.csv).
# Structurally different flag: stillness at a flagged facility, not movement.
# Only the ESCALATING episodes (trigger prolonged_dwell_at_flagged_facility)
# are surfaced on the map.
# --------------------------------------------------------------------------
WINDOW_LOITER_LABEL = {
    "Qatar control (Sep 2025)": "qatar_control_sept2025",
    "Hormuz crisis (Mar 2026)": "hormuz_crisis_mar2026",
    "Hormuz control (Jan 2026)": "hormuz_control_mar2026",
    "Jamnagar / Vadinar — Nayara refinery (Feb 2026)": "jamnagar_vadinar_feb2026",
}
LOITER_RGB = [181, 23, 158]   # magenta — deliberately outside the 4-class palette

# --------------------------------------------------------------------------
# Key locations for the sidebar fly-to navigation. Each carries the camera
# target (centre + zoom), the window it belongs to, and the contextual info
# card shown on arrival. Facilities are drawn as persistent ring markers.
# --------------------------------------------------------------------------
LOCATIONS = {
    "qatar": dict(
        label="Qatar · Ras Laffan (Oct 2025)",
        window="Qatar incident (Oct 2025)",
        center=(25.78, 51.75), zoom=8.3,
        context="Ras Laffan, northern Qatar — one of the world's largest LNG export terminals.",
        finding="GNSS spoofing incident, 4–6 Oct 2025: a navigation halt at Ras Laffan, "
                "reported positions pulled off the berth.",
        facilities=[("Ras Laffan", 25.9061, 51.5992), ("Doha", 25.2969, 51.5511)],
    ),
    "hormuz": dict(
        label="Strait of Hormuz (Mar 2026 crisis)",
        window="Hormuz crisis (Mar 2026)",
        center=(26.45, 56.45), zoom=7.7,
        context="The 40 km chokepoint between Oman and Iran carrying roughly a fifth of seaborne oil.",
        finding="Feb–Mar 2026 crisis window: site of the LENORE finding — a sanctioned tanker "
                "with a physically impossible 82 kn position jump (freeze-then-teleport spoof).",
        facilities=[("Strait of Hormuz", 26.5667, 56.2500)],
    ),
    "mangalore": dict(
        label="New Mangalore (Sep–Oct 2025)",
        window="India baseline / New Mangalore (Sep-Oct 2025)",
        center=(12.95, 74.80), zoom=9.1,
        context="West-coast India deep-water port; SELENIA's documented India route runs through its approaches.",
        finding="SELENIA's documented India route; two detection windows tested clean after a "
                "rigorous SAR↔AIS re-link audit (5 GFW mislinks corrected, no spoofing).",
        facilities=[("New Mangalore port", 12.9200, 74.8000)],
    ),
    "jamnagar": dict(
        label="Jamnagar / Vadinar (Feb 2026)",
        window="Jamnagar / Vadinar — Nayara refinery (Feb 2026)",
        center=(22.50, 69.70), zoom=9.5,
        context="Gulf of Kutch: the EU/UK-sanctioned Nayara Energy (Vadinar) refinery and the "
                "Reliance (Sikka) marine terminal.",
        finding="TIBURON and SEASONS I confirmed shadow-fleet tankers; SEASONS I's 9-day anchorage "
                "loiter was caught by the new dwell detector — invisible to every movement-based check.",
        facilities=[("Vadinar / Nayara refinery", 22.55, 69.65),
                    ("Jamnagar (Sikka) terminal", 22.43, 69.72)],
    ),
}

# Region membership. A classification-percentage comparison is only
# scientifically valid WITHIN a region: same waters, same calibrated
# thresholds, an incident window vs. its own control. Cross-region pairs use
# independently-calibrated cutoffs and are handled separately (threshold
# values only, never raw activity mix).
WINDOW_REGION = {
    "Qatar incident (Oct 2025)": "qatar",
    "Qatar control (Sep 2025)": "qatar",
    "Hormuz crisis (Mar 2026)": "hormuz",
    "Hormuz control (Jan 2026)": "hormuz",
    "India baseline / New Mangalore (Sep-Oct 2025)": "india",
    "Jamnagar / Vadinar — Nayara refinery (Feb 2026)": "india",
}
REGION_LABEL = {"qatar": "Qatar", "hormuz": "Strait of Hormuz", "india": "India west coast"}


def same_region_windows(wname):
    """Windows in the same region as `wname`, excluding it -- the only
    scientifically valid partners for a classification-percentage comparison."""
    reg = WINDOW_REGION.get(wname)
    return [w for w in WINDOWS if w != wname and WINDOW_REGION.get(w) == reg]


# Region -> already-computed threshold-calibration JSON (calibrate_thresholds.py).
CALIB_FILE = {
    "gulf": f"{PROC}/threshold_calibration_hormuz_control.json",
    "india": f"{PROC}/threshold_calibration_india_mangalore.json",
}
# India's Tukey far-out fence (Q3 + 3*IQR) on the confirmed-pair distance
# distribution -- the value inspect_india_tail_outliers.py used to isolate the
# 5 misattributed tail pairs.
INDIA_TAIL_FENCE_KM = 27.48

# Externally-confirmed vessels -- mirrors data/processed/vessel_age_cache.json
# and CONFIRMED_VESSEL_TYPES in src/vessel_history.py (registry lookups done
# this project: MarineTraffic / VesselFinder / vesseltracker / MyShipTracking).
EXT_CONFIRMED = {
    511101414: dict(name="SELENIA", type="asphalt/bitumen tanker", imo="9286437", built=2004),
    352003690: dict(name="HAKKAISAN", type="crude oil tanker (VLCC, LOA 333 m)", imo="9376878", built=2009),
    636018010: dict(name="PATRIS", type="LNG carrier", imo="9766889", built=2018),
    636025162: dict(name="OCEAN CENTURY", type="tug (LOA ~37 m)", imo="9435650", built=2007),
    306531000: dict(name="LENORE", type="crude / products tanker", imo="9259367", built=2004),
}

BUGS = [
    ("Wrong timestamp field (Qatar)", "fetch_ais.py",
     "Positioned on entry/exitTimestamp (= query-range bounds), so every grid cell matched every query time."),
    ("Spoofing hidden in “unmatched”", "match.py",
     "A capped nearest-AIS search dumped far-away activity into an unmatched bucket — severe spoofing read as AIS silence."),
    ("Vessel-type bias in scoring", "vessel_history.py (v2)",
     "Score leaned on vessel type, penalising whole categories instead of observed behaviour."),
    ("FOC list over-broadness", "vessel_history.py (v3)",
     "Flag-of-convenience matching was too permissive, inflating the FOC component for ordinary flags."),
    ("Misleading deviation_km on long gaps", "route_plausibility.py",
     "A fixed km cutoff over a 20 h AIS gap flagged ordinary route flex; moved to sqrt(gap) scaling."),
    ("GFW SAR↔AIS mis-attribution", "relink_sar_ais.py",
     "GFW's given MMSI was wrong by tens of km in dense coastal traffic; added verify-then-repair re-linking."),
]


# --------------------------------------------------------------------------
# Loaders (pure pandas; cached under Streamlit, plain functions otherwise)
# --------------------------------------------------------------------------
def _hav_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_scores():
    if not os.path.exists(SCORES_FILE):
        return None
    return pd.read_csv(SCORES_FILE)


def load_window(name):
    """Return a normalised DataFrame with columns: lat, lon, cls (canonical),
    disp_name, disp_mmsi, val, val_label, extra (dict per row for popup)."""
    cfg = WINDOWS[name]
    p = f"{PROC}/{cfg['file']}"
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    out = pd.DataFrame()
    out["lat"] = pd.to_numeric(df.get("sar_lat"), errors="coerce")
    out["lon"] = pd.to_numeric(df.get("sar_lon"), errors="coerce")

    if cfg["kind"] == "classified":
        out["cls"] = df["classification"].astype(str)
        out["disp_name"] = df["ship_name"] if "ship_name" in df.columns else None
        mm = (pd.to_numeric(df["matched_mmsi"], errors="coerce")
              if "matched_mmsi" in df.columns else pd.Series(np.nan, index=df.index))
        m2 = (pd.to_numeric(df["mmsi"], errors="coerce")
              if "mmsi" in df.columns else pd.Series(np.nan, index=df.index))
        out["disp_mmsi"] = mm.fillna(m2)
        out["val"] = pd.to_numeric(df.get("distance_km"), errors="coerce")
        out["val_label"] = "nearest-AIS distance (km)"
        # the matched-AIS position that the SAR return was compared against --
        # present in every *_classified / *_reclassified file (all windows).
        # This is what the SAR<->AIS connector line is drawn from; it was being
        # dropped here before, which is why the Gulf/Qatar maps showed no links.
        out["ais_lat"] = pd.to_numeric(df.get("ais_lat"), errors="coerce")
        out["ais_lon"] = pd.to_numeric(df.get("ais_lon"), errors="coerce")
    else:  # India re-linking layer
        out["cls"] = df["link_status"].map(RELINK_TO_CANON).fillna("unmatched")
        out["disp_name"] = df["relinked_name"].fillna(df["gfw_ship_name"])
        out["disp_mmsi"] = df["relinked_mmsi"].fillna(df["gfw_mmsi"])
        out["val"] = pd.to_numeric(df["relinked_dist_km"], errors="coerce")
        out["val_label"] = "re-linked distance (km)"
        out["link_status"] = df["link_status"]
        out["change_vs_gfw"] = df["change_vs_gfw"]
        # no single matched-AIS point in the re-link layer -- the SAR<->AIS
        # before/after lines for India live in the dedicated comparison panel.
        out["ais_lat"] = np.nan
        out["ais_lon"] = np.nan

    out = out.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    return out


def load_deep_history(mmsi):
    p = f"{PROC}/vessel_deep_history_{int(mmsi)}.csv"
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    d["start"] = pd.to_datetime(d["start"], utc=True, errors="coerce")
    d["end"] = pd.to_datetime(d["end"], utc=True, errors="coerce")
    return d


def load_loiter(loiter_label):
    """Escalating dwell episodes for a window: the rows of
    data/processed/loiter_flags_{label}.csv whose trigger is
    `prolonged_dwell_at_flagged_facility` (escalate == True). No new data."""
    if not loiter_label:
        return None
    p = f"{PROC}/loiter_flags_{loiter_label}.csv"
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    d = d[d["escalate"] == True]  # noqa: E712
    if d.empty:
        return d
    d["centroid_lat"] = pd.to_numeric(d["centroid_lat"], errors="coerce")
    d["centroid_lon"] = pd.to_numeric(d["centroid_lon"], errors="coerce")
    d["duration_h"] = pd.to_numeric(d["duration_h"], errors="coerce")
    return d.dropna(subset=["centroid_lat", "centroid_lon"]).reset_index(drop=True)


def load_raw_ais(label):
    """Already-cached raw AIS positions for a window (no fetch). Slim columns."""
    if not label:
        return None
    p = f"{RAW}/{label}_ais_positions.csv"
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p, usecols=lambda c: c in ("mmsi", "timestamp", "lat", "lon", "ship_name"))
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    d["lat"] = pd.to_numeric(d["lat"], errors="coerce")
    d["lon"] = pd.to_numeric(d["lon"], errors="coerce")
    d["mmsi"] = pd.to_numeric(d["mmsi"], errors="coerce")
    return (d.dropna(subset=["mmsi", "timestamp", "lat", "lon"])
             .astype({"mmsi": "int64"})
             .sort_values("timestamp")
             .reset_index(drop=True))


def vessel_tracks(label):
    """dict: mmsi -> DataFrame(timestamp, lat, lon) for that vessel, time-sorted.

    This is the sequence of AIS positions ALREADY in the pulled window -- the
    connected historical track drawn behind each SAR detection."""
    d = load_raw_ais(label)
    if d is None:
        return {}
    return {int(m): g[["timestamp", "lat", "lon"]].reset_index(drop=True)
            for m, g in d.groupby("mmsi")}


def nearest_ais_pos(tracks, mmsi, ts):
    """(lat, lon, dt_minutes) of a vessel's cached AIS fix nearest to `ts`."""
    if not tracks or pd.isna(mmsi):
        return None
    g = tracks.get(int(mmsi))
    if g is None or g.empty:
        return None
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    i = (g["timestamp"] - ts).abs().idxmin()
    r = g.loc[i]
    return float(r["lat"]), float(r["lon"]), abs((r["timestamp"] - ts).total_seconds()) / 60.0


def _bearing_deg(lat1, lon1, lat2, lon2):
    """Initial great-circle bearing lat1,lon1 -> lat2,lon2 in degrees (0=N)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


# --------------------------------------------------------------------------
# deck.gl / pydeck map helpers (replaces the old Folium/Leaflet renderer --
# rendering technology only; same data, classes and interactions).
# --------------------------------------------------------------------------
_SHIP_ATLAS_CACHE = {}


def ship_icon_atlas():
    """A single white ship silhouette (RGBA, 128px) as a PNG data URI, used as
    the IconLayer atlas. `mask: true` in the icon mapping means deck.gl tints
    this silhouette with each vessel's classification colour, so one asset
    serves all four buckets. Bow points up (native angle 0)."""
    if "uri" in _SHIP_ATLAS_CACHE:
        return _SHIP_ATLAS_CACHE["uri"]
    # tight boat hull, bow up: pointed bow, tapered flat stern.
    hull = [(64, 14), (86, 52), (88, 100), (78, 116), (50, 116), (40, 100), (42, 52)]
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.polygon(hull, fill=(255, 255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        pts = " ".join(f"{x},{y}" for x, y in hull)
        svg = ("<svg xmlns='http://www.w3.org/2000/svg' width='128' height='128'>"
               f"<polygon points='{pts}' fill='white'/></svg>")
        uri = "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
    _SHIP_ATLAS_CACHE["uri"] = uri
    return uri


def ship_icon_obj(atlas=None):
    """Per-row IconLayer icon spec (deck.gl reads this dict straight from the
    data, so pydeck never mistakes the data URI for an accessor expression).
    `mask: true` -> tinted by the row's getColor."""
    return {"url": atlas or ship_icon_atlas(), "width": 128, "height": 128,
            "anchorX": 64, "anchorY": 64, "mask": True}


def deck_basemap():
    """(map_provider, map_style, label). Mapbox dark-v11 when a token is
    configured (env MAPBOX_API_KEY / MAPBOX_TOKEN or st.secrets['mapbox']);
    otherwise the tokenless CARTO dark-matter style."""
    tok = os.environ.get("MAPBOX_API_KEY") or os.environ.get("MAPBOX_TOKEN")
    if not tok:
        try:
            import streamlit as st
            tok = (st.secrets.get("mapbox", {}) or {}).get("token")
        except Exception:
            tok = None
    if tok:
        return "mapbox", "mapbox://styles/mapbox/dark-v11", "Mapbox dark-v11", tok
    return "carto", "dark", "CARTO dark-matter (no token)", None


_TRAIL_MAX_PTS = 48


def _downsample(arr, n=_TRAIL_MAX_PTS):
    if len(arr) > n:
        arr = arr[:: int(np.ceil(len(arr) / n))]
    return arr


def _cut_trail(g, hours):
    """Keep only the last `hours` of a time-sorted track (hours falsy -> all)."""
    if g is None or g.empty or not hours or hours <= 0:
        return g
    cutoff = g["timestamp"].max() - pd.Timedelta(hours=float(hours))
    return g[g["timestamp"] >= cutoff]


def gradient_segments(path, rgb, a_lo=18, a_hi=255):
    """Split an ordered [[lon,lat], ...] path into per-segment records whose
    alpha ramps a_lo -> a_hi oldest to newest, so the trail dissolves into the
    basemap behind the vessel and is brightest at its current position."""
    segs, m = [], len(path)
    for i in range(m - 1):
        a = int(a_lo + (a_hi - a_lo) * ((i + 1) / (m - 1)))
        segs.append({"path": [path[i], path[i + 1]], "color": rgb + [a]})
    return segs


def _dash_segments(p0, p1, n=13):
    """A straight p0->p1 line rendered as `n`-slot on/off dashes (list of the
    'on' sub-segments) -- a portable stand-in for PathStyleExtension dashing."""
    out = []
    for i in range(0, n, 2):
        f0, f1 = i / n, min((i + 1) / n, 1.0)
        out.append({"path": [
            [p0[0] + (p1[0] - p0[0]) * f0, p0[1] + (p1[1] - p0[1]) * f0],
            [p0[0] + (p1[0] - p0[0]) * f1, p0[1] + (p1[1] - p0[1]) * f1]]})
    return out


def is_flagged_row(cls, mmsi_i, scores_mmsi, loiter_mmsi):
    """A detection worth showing in the minimal default view: a spoof/discrepant
    class, a vessel on the v6 flagged leaderboard, or a dwell escalation."""
    return (cls in ("likely_spoofed", "discrepant")
            or (mmsi_i is not None and mmsi_i in scores_mmsi)
            or (mmsi_i is not None and mmsi_i in loiter_mmsi))


def build_map_records(df, tracks, kind, scores_mmsi, loiter_mmsi, *,
                      trail_hours=18, flagged_only=True, show_unmatched=False,
                      atlas=None):
    """Reshape a normalised window DataFrame + per-MMSI track dict into deck.gl
    record lists. Applies the declutter rules: unmatched hidden unless asked,
    only flagged/escalated vessels in the minimal view, trails capped to the
    last `trail_hours` with an opacity gradient. No new data."""
    icon_obj = ship_icon_obj(atlas)
    icons, trail_segs, trail_glow, recent_dots = [], [], [], []
    conn_segs, ais_dots = [], []
    full_tracks = {}
    has_ais = "ais_lat" in df.columns and "ais_lon" in df.columns
    for _, r in df.iterrows():
        cls = r["cls"]
        mmsi_i = None if pd.isna(r["disp_mmsi"]) else int(float(r["disp_mmsi"]))
        if cls == "unmatched":
            if not show_unmatched:      # gray background traffic: off by default
                continue
        elif flagged_only and not is_flagged_row(cls, mmsi_i, scores_mmsi, loiter_mmsi):
            continue                    # minimal view: flagged / escalated only
        rgb = CLS_RGB.get(cls, [136, 136, 136])
        nm = r["disp_name"] if pd.notna(r["disp_name"]) else "(no AIS name)"
        val = "n/a" if pd.isna(r["val"]) else f"{r['val']:.2f} km"

        # SAR return <-> its matched-AIS position: the connector line. Drawn for
        # every classified window (Qatar, Hormuz, Jamnagar); its length IS the
        # distance_km that drives the classification, so a long red line is a
        # likely_spoofed detection made visible.
        if has_ais and pd.notna(r["ais_lat"]) and pd.notna(r["ais_lon"]):
            d_deg = abs(float(r["ais_lat"]) - float(r["lat"])) + abs(float(r["ais_lon"]) - float(r["lon"]))
            if d_deg > 1e-4:            # skip sub-100 m matched pairs (invisible)
                sar_xy = [float(r["lon"]), float(r["lat"])]
                ais_xy = [float(r["ais_lon"]), float(r["ais_lat"])]
                conn_segs.append({"path": [sar_xy, ais_xy], "color": rgb + [215],
                                  "glow": rgb + [46], "cls": cls})
                ais_dots.append({
                    "position": ais_xy, "color": rgb + [235],
                    "name": f"{nm} — reported AIS position",
                    "mmsi": ("—" if mmsi_i is None else str(mmsi_i)),
                    "cls": f"matched AIS fix ({cls})",
                    "val_label": r["val_label"], "val": val, "link_html": ""})

        heading = 0.0
        tk = tracks.get(mmsi_i) if mmsi_i is not None else None
        if tk is not None and len(tk) >= 2:
            full = _downsample(tk[["lon", "lat"]].to_numpy()).tolist()
            if mmsi_i is not None:
                full_tracks[mmsi_i] = full
            cut = _cut_trail(tk, trail_hours)
            pth = (_downsample(cut[["lon", "lat"]].to_numpy()).tolist()
                   if cut is not None and len(cut) >= 2 else full[-2:])
            (lo1, la1), (lo2, la2) = pth[-2], pth[-1]
            heading = _bearing_deg(la1, lo1, la2, lo2)
            trail_segs.extend(gradient_segments(pth, rgb))
            g0 = max(0, int(len(pth) * 0.6))
            trail_glow.append({"path": pth[g0:], "color": rgb + [50]})
            recent_dots.append({"position": pth[-1], "color": rgb + [255]})

        link_html = ""
        if kind == "relinked" and "link_status" in r:
            link_html = (f"<br/>link_status: <b>{r['link_status']}</b>"
                         f"<br/>vs GFW: {r['change_vs_gfw']}")
        icons.append({
            "position": [float(r["lon"]), float(r["lat"])],
            "name": nm, "mmsi": ("—" if mmsi_i is None else str(mmsi_i)),
            "mmsi_i": mmsi_i, "cls": cls,
            "color": rgb, "halo": rgb + [55],
            "angle": float(-heading % 360.0),
            "val": val, "val_label": r["val_label"], "link_html": link_html,
            "icon": icon_obj,
        })
    return dict(icons=icons, trail_segs=trail_segs, trail_glow=trail_glow,
                recent_dots=recent_dots, full_tracks=full_tracks,
                conn_segs=conn_segs, ais_dots=ais_dots)


def facility_records(specs):
    """[(name, lat, lon), ...] -> tooltip-compatible marker records."""
    return [{"position": [float(lon), float(lat)], "name": name,
             "mmsi": "—", "cls": "facility", "val_label": "role",
             "val": "reference facility", "link_html": ""}
            for name, lat, lon in specs]


def loiter_records(dfl):
    """Escalating-dwell rows -> pulsing-marker records (position + labels)."""
    if dfl is None or dfl.empty:
        return []
    out = []
    for _, r in dfl.iterrows():
        nm = r["ship_name"] if pd.notna(r.get("ship_name")) else f"MMSI {int(r['mmsi'])}"
        dur_h = float(r["duration_h"])
        out.append({
            "position": [float(r["centroid_lon"]), float(r["centroid_lat"])],
            "name": nm, "mmsi": str(int(r["mmsi"])), "cls": "prolonged dwell",
            "label": f"[DWELL] {nm}",
            "val_label": "dwell", "link_html": "",
            "val": f"{dur_h:.0f} h ({dur_h / 24:.1f} d) @ {r['nearest_facility']}",
        })
    return out


def build_map_deck(rec, facilities, loiters, view, *, detail="individual",
                   picked_mmsi=None):
    """Assemble the dark GEOINT deck. `view` is a dict(latitude, longitude,
    zoom); a transition_duration is always attached so any change of `view`
    from the sidebar fly-to list animates as a smooth camera move."""
    provider, style, _, tok = deck_basemap()
    layers = []

    # --- persistent reference facilities ---------------------------------
    if facilities:
        layers.append(pdk.Layer(
            "ScatterplotLayer", data=facilities, id="fac_ring",
            get_position="position", filled=False, stroked=True,
            get_line_color=[228, 232, 238, 220], line_width_min_pixels=2,
            get_radius=10, radius_min_pixels=9, radius_max_pixels=15,
            pickable=True))
        layers.append(pdk.Layer(
            "TextLayer", data=facilities, id="fac_txt",
            get_position="position", get_text="name", get_size=11,
            get_color=[214, 221, 231, 220], get_pixel_offset=[0, -17],
            size_min_pixels=10, size_max_pixels=13))

    # --- loiter / dwell escalations (structurally distinct flag) ---------
    if loiters:
        for rad, alpha in ((36, 38), (22, 90), (10, 235)):
            layers.append(pdk.Layer(
                "ScatterplotLayer", data=loiters, id=f"loiter_{rad}",
                get_position="position", get_fill_color=LOITER_RGB + [alpha],
                get_radius=rad, radius_min_pixels=rad, radius_max_pixels=rad + 8,
                stroked=False, pickable=(rad == 10)))
        layers.append(pdk.Layer(
            "TextLayer", data=loiters, id="loiter_txt",
            get_position="position", get_text="label", get_size=12,
            get_color=[236, 173, 224, 255], get_pixel_offset=[0, 16],
            size_min_pixels=10, size_max_pixels=14))

    if detail == "clustered" and rec["icons"]:
        # low-zoom density aggregation instead of hundreds of overlapping icons
        layers.append(pdk.Layer(
            "HexagonLayer", data=rec["icons"], id="density",
            get_position="position", radius=2600, coverage=0.82,
            extruded=False, pickable=False, opacity=0.38,
            color_range=[[38, 52, 66], [51, 80, 104], [74, 121, 140],
                         [122, 168, 150], [201, 189, 118], [233, 196, 106]]))
    else:
        # SAR<->AIS connector lines (class-coloured): glow then core
        if rec.get("conn_segs"):
            layers.append(pdk.Layer(
                "PathLayer", data=rec["conn_segs"], id="conn_glow",
                get_path="path", get_color="glow", get_width=6,
                width_min_pixels=5, cap_rounded=True, joint_rounded=True,
                pickable=False))
            layers.append(pdk.Layer(
                "PathLayer", data=rec["conn_segs"], id="conn_core",
                get_path="path", get_color="color", get_width=2,
                width_min_pixels=1.6, cap_rounded=True, joint_rounded=True,
                pickable=False))
        if rec.get("ais_dots"):
            layers.append(pdk.Layer(
                "ScatterplotLayer", data=rec["ais_dots"], id="ais_dots",
                get_position="position", filled=False, stroked=True,
                get_line_color="color", line_width_min_pixels=1.5,
                get_radius=3, radius_min_pixels=3, radius_max_pixels=6,
                pickable=True))
        if rec["trail_glow"]:
            layers.append(pdk.Layer(
                "PathLayer", data=rec["trail_glow"], id="trail_glow",
                get_path="path", get_color="color", get_width=7,
                width_min_pixels=6, cap_rounded=True, joint_rounded=True,
                pickable=False))
        if rec["trail_segs"]:
            layers.append(pdk.Layer(
                "PathLayer", data=rec["trail_segs"], id="trail_grad",
                get_path="path", get_color="color", get_width=2,
                width_min_pixels=1.4, cap_rounded=True, pickable=False))
        if picked_mmsi is not None and picked_mmsi in rec["full_tracks"]:
            layers.append(pdk.Layer(
                "PathLayer", data=[{"path": rec["full_tracks"][picked_mmsi]}],
                id="full_track", get_path="path", get_color=[255, 255, 255, 150],
                get_width=1.3, width_min_pixels=1, cap_rounded=True, pickable=False))
        if rec["recent_dots"]:
            layers.append(pdk.Layer(
                "ScatterplotLayer", data=rec["recent_dots"], id="recent",
                get_position="position", get_fill_color="color",
                get_radius=3.2, radius_min_pixels=3, radius_max_pixels=7,
                stroked=True, get_line_color=[8, 12, 18, 220],
                line_width_min_pixels=1, pickable=False))
        layers.append(pdk.Layer(
            "IconLayer", data=rec["icons"], id="ships",
            get_icon="icon", get_position="position", get_color="color",
            get_angle="angle", get_size=15, size_min_pixels=8, size_max_pixels=22,
            billboard=True, pickable=True))

    vs = pdk.ViewState(latitude=view["latitude"], longitude=view["longitude"],
                       zoom=view["zoom"], pitch=0, bearing=0,
                       transition_duration=view.get("dur", 2400))
    tooltip = {
        "html": "<b>{name}</b><br/>MMSI: {mmsi}<br/>class: <b>{cls}</b>"
                "<br/>{val_label}: {val}{link_html}",
        "style": {"backgroundColor": "#0b0f14", "color": "#e6edf3",
                  "fontSize": "12px", "border": "1px solid #30363d"},
    }
    return pdk.Deck(layers=layers, initial_view_state=vs,
                    map_provider=provider, map_style=style,
                    api_keys=({"mapbox": tok} if tok else None),
                    tooltip=tooltip)


def _map_legend_html(kind, show_unmatched):
    """A persistent, always-visible legend for every marker/colour/line on the
    map -- not a hover tooltip. Rendered for all windows; the re-link line-style
    rows are only relevant to the India re-linking layer."""
    def sw(color):   # colour swatch
        return (f"<span style='display:inline-block;width:11px;height:11px;"
                f"background:{color};border:1px solid #00000055;vertical-align:middle;"
                f"margin-right:5px'></span>")

    def line(css):   # line-style sample
        return (f"<span style='display:inline-block;width:22px;height:0;"
                f"border-top:{css};vertical-align:middle;margin-right:5px'></span>")

    loiter = f"rgb({LOITER_RGB[0]},{LOITER_RGB[1]},{LOITER_RGB[2]})"
    rows = [
        sw(CLS_COLORS["matched"]) + "matched",
        sw(CLS_COLORS["discrepant"]) + "discrepant",
        sw(CLS_COLORS["likely_spoofed"]) + "likely_spoofed",
        sw(CLS_COLORS["unmatched"]) + "unmatched "
        + ("(shown)" if show_unmatched else "(off by default)"),
        (f"<span style='display:inline-block;width:11px;height:11px;border-radius:50%;"
         f"border:2px solid {loiter};vertical-align:middle;margin-right:5px'></span>"
         "loiter-flagged (prolonged dwell)"),
        ("<span style='display:inline-block;width:11px;height:11px;border:1px solid #e4e8ee;"
         "vertical-align:middle;margin-right:5px'></span>reference facility"),
        line("2px solid #cdd6df") + "SAR ↔ matched-AIS connector (class-coloured)",
    ]
    if kind == "relinked":
        rows += [
            line("2px dashed #9696a0") + "GFW original link",
            line("3px solid #2a9d8f") + "corrected Hungarian re-link",
        ]
    cells = "".join(
        f"<span style='display:inline-block;margin:2px 16px 2px 0;white-space:nowrap'>{r}</span>"
        for r in rows)
    return ("<div style='font-size:12px;line-height:1.9;padding:8px 10px;"
            "border:1px solid #30363d;border-radius:6px;background:#0b0f14;"
            f"margin-bottom:6px'><b style='color:#8b98a5'>LEGEND</b> &nbsp; {cells}</div>")


def _badge(tag, body, colour):
    """A flat status badge: an uppercase colour-coded tag + text, no icon --
    replaces st.success/st.info callouts for a GEOINT-tool tone."""
    return (f"<div style='border:1px solid #30363d;border-left:3px solid {colour};"
            f"border-radius:5px;padding:8px 11px;margin:4px 0;background:#0b0f14'>"
            f"<span style='font-size:10px;letter-spacing:1px;font-weight:700;"
            f"color:{colour}'>{tag}</span><br/>"
            f"<span style='font-size:12.5px;color:#e6edf3'>{body}</span></div>")


def render_vessel_detail(st, scores, row, deep_loader):
    """The vessel detail panel -- score breakdown, why-flagged sentence,
    external-confirmation line, deep-history summary, LENORE figure.
    Shared by the drill-down tab and the map click-to-inspect."""
    mmsi = int(row["mmsi"])

    c1, c2 = st.columns([3, 2])
    with c1:
        fig, ax = plt.subplots(figsize=(6, 1.7))
        b = float(row["behavioral_score_adjusted"])
        f = float(row["foc_score"])
        a = float(row["age_score"])
        for val, left, col, lab in [
                (b, 0, "#264653", "behavioral"),
                (f, b, "#e9c46a", "FOC"),
                (a, b + f, "#8ab17d", "age")]:
            ax.barh([0], [val], left=[left], color=col, label=lab)
            if val > 0:
                ax.text(left + val / 2, 0, f"{val:g}", ha="center",
                        va="center", color="white", fontweight="bold", fontsize=9)
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xlabel("reliability score (0–100)")
        ax.set_title(f"{row['ship_name']}  —  total {row['reliability_score']:g} / 100",
                     fontsize=11, fontweight="bold")
        ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.55),
                  frameon=False, fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        st.metric("Fleet rank (non-artifact)", _rank(scores, mmsi))
        st.write(f"**Flag:** {row.get('flag_state', '—')}  ·  "
                 f"**FOC tier:** {row.get('foc_list_matched', '—')}")
        st.write(f"**Flagged events:** {int(row['flagged_event_count'])}  "
                 f"in `{row['appears_in_windows']}`")
        st.write(f"**Types:** {row['flagged_event_types']}")

    st.markdown("**Why flagged**")
    st.markdown(_badge("WHY FLAGGED", why_flagged_sentence(row), "#e9c46a"),
                unsafe_allow_html=True)

    if mmsi in EXT_CONFIRMED:
        e = EXT_CONFIRMED[mmsi]
        st.markdown(_badge(
            "VERIFIED — EXTERNAL REGISTRY",
            f"{e['type']} · IMO {e['imo']} · built {e['built']} "
            "(registry cross-check; see vessel_age_cache.json)", "#2a9d8f"),
            unsafe_allow_html=True)
    else:
        src = row.get("vessel_type_source", "gfw_identity")
        st.markdown(_badge(
            "TYPE — GFW-NATIVE (NO EXTERNAL OVERRIDE)",
            f"{row.get('resolved_vessel_type', '—')}  (source: {src})", "#8b98a5"),
            unsafe_allow_html=True)

    dh = deep_loader(mmsi)
    if dh is None:
        st.caption("No 6-month deep-history file for this vessel.")
    else:
        _render_deep(st, dh)
        if mmsi == 306531000:
            st.markdown("#### LENORE — speed anomaly")
            p = f"{FIGS}/lenore_speed_anomaly.png"
            if os.path.exists(p):
                st.image(p, caption="figures/lenore_speed_anomaly.png — pre-generated "
                         "by src/generate_figures.py, embedded as-is (not recomputed "
                         "in the dashboard).")
            else:
                st.warning(f"`{p}` not found.")


def load_calibration(region):
    """Parsed threshold_calibration_{region}.json -- the numbers
    calibrate_thresholds.py already derived. No recomputation here."""
    p = CALIB_FILE.get(region)
    if not p or not os.path.exists(p):
        return None
    with open(p) as fh:
        j = json.load(fh)
    dist = j["sar_ais_distance_calibration"]
    return {
        "region": j.get("region", region),
        "confirmed_pct": dist["confirmed_pair_percentiles_km"],
        "pre_screen_pct": dist.get("pre_screen_confirmed_pair_percentiles_km"),
        "matched_km": j["derived_thresholds"]["matched_km"],          # p50
        "likely_spoofed_km": j["derived_thresholds"]["likely_spoofed_km"],  # p99
        "p95_km": dist["likely_spoofed_tail_check"]["p95_km"],
        "n_confirmed": dist["n_confirmed_identity_pairs"],
        "n_pre_screen": dist["n_confirmed_identity_pairs_pre_screen"],
        "n_excluded": dist["misattribution_screen"]["n_excluded_suspected_misattribution"],
        "confidence": dist.get("confidence"),
    }


def gulf_confirmed_pair_distances():
    """REAL per-pair distances: distance_km for the confirmed identity pairs in
    hormuz_control_mar2026_classified.csv (the pre-screen set the Gulf
    calibration is taken over -- non-null count matches n_pre_screen)."""
    p = f"{PROC}/hormuz_control_mar2026_classified.csv"
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    v = pd.to_numeric(d.get("distance_km"), errors="coerce").dropna().to_numpy()
    return v if len(v) else None


def reconstruct_dist_from_percentiles(pct, n=3000):
    """Approximate sample drawn from a stored percentile grid by inverse-CDF
    interpolation. Used for India, where calibrate_thresholds.py persists only
    p10..p99.9 of the confirmed-pair distances, not the raw values."""
    grid = [("p10", 10), ("p25", 25), ("p50", 50), ("p75", 75),
            ("p90", 90), ("p95", 95), ("p99", 99), ("p99.9", 99.9)]
    xs = [0.0] + [float(pct[k]) for k, _ in grid if k in pct]
    ps = [0.0] + [v / 100.0 for k, v in grid if k in pct]
    xs = np.maximum.accumulate(xs)  # keep the CDF monotone
    u = np.linspace(0.002, 0.998, n)
    return np.interp(u, ps, xs)


def india_gfw_corrected_pairs():
    """The SAR detections where the Hungarian re-linker OVERRODE GFW's MMSI
    (change_vs_gfw == 'gfw_corrected' in india_mangalore_relinked.csv).

    For each: the SAR point and the corrected re-link distance are read straight
    from the CSV; GFW's original vessel position is reconstructed by nearest-time
    lookup in the cached raw India AIS file (its separation is not persisted).
    `tail_outlier` marks the pairs whose GFW separation clears India's Tukey
    far-out fence -- the ones inspect_india_tail_outliers.py isolated."""
    p = f"{PROC}/india_mangalore_relinked.csv"
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p, parse_dates=["sar_timestamp"])
    g = d[d["change_vs_gfw"] == "gfw_corrected"].copy()
    if g.empty:
        return g
    tr = vessel_tracks("india_mangalore_sep_oct2025")
    rows = []
    for _, r in g.iterrows():
        gm = int(r["gfw_mmsi"]) if pd.notna(r["gfw_mmsi"]) else None
        rm = int(r["relinked_mmsi"]) if pd.notna(r["relinked_mmsi"]) else None
        gp = nearest_ais_pos(tr, gm, r["sar_timestamp"]) if gm else None
        rp = nearest_ais_pos(tr, rm, r["sar_timestamp"]) if rm else None
        gd = (_hav_km(float(r["sar_lat"]), float(r["sar_lon"]), gp[0], gp[1])
              if gp else np.nan)
        rows.append(dict(
            sar_id=int(r["sar_id"]), sar_ts=r["sar_timestamp"],
            sar_lat=float(r["sar_lat"]), sar_lon=float(r["sar_lon"]),
            gfw_mmsi=gm, gfw_name=r.get("gfw_ship_name"),
            gfw_lat=(gp[0] if gp else np.nan), gfw_lon=(gp[1] if gp else np.nan),
            gfw_dt_min=(gp[2] if gp else np.nan), gfw_dist_km=float(gd) if pd.notna(gd) else np.nan,
            relinked_mmsi=rm, relinked_name=r.get("relinked_name"),
            rel_lat=(rp[0] if rp else np.nan), rel_lon=(rp[1] if rp else np.nan),
            rel_dt_min=(rp[2] if rp else np.nan),
            relinked_dist_km=pd.to_numeric(r.get("relinked_dist_km"), errors="coerce"),
            tail_outlier=bool(pd.notna(gd) and gd > INDIA_TAIL_FENCE_KM),
        ))
    return pd.DataFrame(rows)


def _spoof_distance_for(mmsi):
    """Max likely_spoofed SAR<->AIS distance recorded for an MMSI across the
    already-classified window files (for the 'why flagged' sentence)."""
    for fn in ("hormuz_crisis_mar2026_reclassified.csv",
               "hormuz_control_mar2026_reclassified.csv",
               "qatar_gnss_spoofing_oct2025_classified.csv",
               "qatar_control_sept2025_classified.csv"):
        p = f"{PROC}/{fn}"
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        key = "matched_mmsi" if "matched_mmsi" in d.columns else "mmsi"
        s = d[(pd.to_numeric(d[key], errors="coerce") == mmsi)
              & (d["classification"].astype(str) == "likely_spoofed")]
        v = pd.to_numeric(s.get("distance_km"), errors="coerce").dropna()
        if len(v):
            return float(v.max()), fn
    return None


def why_flagged_sentence(row):
    """Plain-language 'why flagged' string built purely from this vessel's real
    v6 score fields (+ a distance clause from the classified files when the
    vessel actually carries a sar_ais_likely_spoofed event)."""
    ev = str(row.get("flagged_event_types") or "")
    n_ev = row.get("flagged_event_count")
    wins = row.get("appears_in_windows") or "—"
    drivers = []
    if "trajectory_deviation_high" in ev:
        drivers.append("high trajectory-prediction deviation")
    if "genuine_cluster_member" in ev:
        drivers.append("membership of a genuine spatiotemporal cluster")
    if "sar_ais_likely_spoofed" in ev:
        drivers.append("a SAR position its own AIS cannot explain")
    lead = (f"Flagged because: {int(n_ev)} flagged event(s) in `{wins}`"
            + (" — " + "; ".join(drivers) if drivers else "") + ".")

    b = float(row.get("behavioral_score_adjusted") or 0.0)
    f = float(row.get("foc_score") or 0.0)
    a = float(row.get("age_score") or 0.0)
    flag = row.get("flag_state") or "—"
    foc = row.get("foc_list_matched") or "none"
    age = row.get("vessel_age_years")
    age_clause = (f"age={float(age):.0f} yr ({a:g} pts)" if pd.notna(age)
                  else f"age unresolved ({a:g} pts)")
    build = (f"Score: behavioral {b:g} pts; flag={flag} ({foc} list, {f:g} pts); "
             f"{age_clause} → total {float(row['reliability_score']):g}/100.")

    dist_clause = ""
    if "sar_ais_likely_spoofed" in ev:
        hit = _spoof_distance_for(int(row["mmsi"]))
        cal = load_calibration("gulf")
        if hit and cal:
            dist_clause = (f" Distance {hit[0]:.1f} km exceeds the "
                           f"{cal['likely_spoofed_km']:.1f} km likely_spoofed cutoff "
                           f"(`{hit[1]}`).")
    return lead + " " + build + dist_clause


# --------------------------------------------------------------------------
# Data-file audit (used by the sidebar and by `python src/dashboard.py`)
# --------------------------------------------------------------------------
def file_audit():
    expected = [("scoring (v6, FINAL)", SCORES_FILE)]
    for wn, cfg in WINDOWS.items():
        expected.append((f"window: {wn}", f"{PROC}/{cfg['file']}"))
    expected += [
        ("threshold-correction figure", f"{FIGS}/threshold_correction_comparison.png"),
        ("LENORE speed figure", f"{FIGS}/lenore_speed_anomaly.png"),
        ("score-breakdown figure", f"{FIGS}/vessel_score_breakdown.png"),
        ("bugs-timeline figure", f"{FIGS}/bugs_timeline.png"),
        ("two-tier architecture figure", f"{FIGS}/two_tier_architecture.png"),
        ("age cache", f"{PROC}/vessel_age_cache.json"),
        ("v4 scoring (methodology before/after only)", f"{PROC}/vessel_reliability_scores_v4.csv"),
        ("v5 scoring (methodology before/after only)", f"{PROC}/vessel_reliability_scores_v5.csv"),
    ]
    dh = sorted(glob.glob(f"{PROC}/vessel_deep_history_*.csv"))
    expected.append((f"deep-history files ({len(dh)} found)",
                     dh[0] if dh else f"{PROC}/vessel_deep_history_*.csv"))
    return [(label, path, os.path.exists(path)) for label, path in expected]


# ==========================================================================
# Streamlit UI
# ==========================================================================
def run_app():
    import streamlit as st

    st.set_page_config(page_title="Maritime anomaly — retrospective",
                       layout="wide")

    # cache the loaders for this session
    _scores = st.cache_data(load_scores)
    _window = st.cache_data(load_window)
    _deep = st.cache_data(load_deep_history)
    _tracks = st.cache_data(vessel_tracks)
    _calib = st.cache_data(load_calibration)
    _gulf_dists = st.cache_data(gulf_confirmed_pair_distances)
    _india_pairs = st.cache_data(india_gfw_corrected_pairs)
    _atlas = st.cache_data(ship_icon_atlas)
    _loiter = st.cache_data(load_loiter)

    st.title("Maritime anomaly detection — project retrospective")
    st.caption("Ships only. Exploration layer over already-verified outputs — no API calls, no new data. "
               f"Scoring: `{os.path.basename(SCORES_FILE)}` (final, fully audited).")

    scores = _scores()

    # ---------------- sidebar ----------------
    wlist = list(WINDOWS)
    st.session_state.setdefault("wsel", wlist[2])
    st.session_state.setdefault("view", None)
    st.session_state.setdefault("arrived", None)
    st.session_state.setdefault("cur_window", None)

    def _fly_to(loc_key):
        """Sidebar fly-to: switch to the location's window and set the camera
        target. build_map_deck attaches a transition_duration, so Streamlit's
        deck.gl view-state diff animates this as a smooth camera move."""
        loc = LOCATIONS[loc_key]
        seq = st.session_state.get("_navseq", 0) + 1
        st.session_state._navseq = seq
        st.session_state.wsel = loc["window"]
        st.session_state.cur_window = loc["window"]
        st.session_state.arrived = loc_key
        # a per-click transition duration (2.2-2.6 s) — varying it guarantees the
        # deck.gl view-state diff re-applies the transition even right after a
        # manual pan/zoom (which drops the transition prop from live view state).
        st.session_state.view = dict(latitude=loc["center"][0],
                                     longitude=loc["center"][1], zoom=loc["zoom"],
                                     dur=2200 + (seq % 5) * 100)

    st.sidebar.header("Fly to")
    st.sidebar.caption("Animated camera move to a key project location.")
    for _k, _loc in LOCATIONS.items():
        st.sidebar.button(_loc["label"], key=f"nav_{_k}", use_container_width=True,
                          on_click=_fly_to, args=(_k,))

    st.sidebar.divider()
    st.sidebar.header("Window")
    wname = st.sidebar.selectbox("Selected window", wlist, key="wsel")
    _peers = same_region_windows(wname)
    wcompare = st.sidebar.selectbox(
        "Compare against (same-region only)", _peers if _peers else ["(no same-region window)"],
        help="The Compare tab's classification-% chart is only valid within one region "
             "(same waters, same calibrated thresholds). Cross-region is a separate "
             "threshold-only section.")
    if not _peers:
        wcompare = None

    # manual window switch (not via a fly-to button) -> recentre, drop the card
    if st.session_state.cur_window != wname:
        st.session_state.cur_window = wname
        st.session_state.arrived = None
        _d0 = _window(wname)
        if _d0 is not None and not _d0.empty:
            st.session_state.view = dict(latitude=float(_d0["lat"].mean()),
                                         longitude=float(_d0["lon"].mean()), zoom=5.6)

    st.sidebar.divider()
    st.sidebar.header("Map layers")
    show_all = st.sidebar.checkbox("show all classified traffic (not just flagged)",
                                   value=False, key="show_all")
    show_bg = st.sidebar.checkbox("show background traffic", value=False, key="show_bg",
                                  help="include the gray 'unmatched' detections")
    draw_tracks = st.sidebar.checkbox("draw vessel AIS trails", value=True, key="draw_tracks")
    trail_hours = st.sidebar.slider("trail length — hours before latest fix",
                                    6, 120, 18, step=6, key="trail_hours")
    render_mode = st.sidebar.radio(
        "vessel rendering",
        ["Auto (cluster when zoomed out)", "Individual ships", "Clustered density"],
        key="render_mode")

    st.sidebar.divider()
    with st.sidebar.expander("Data sources loaded", expanded=False):
        for label, path, ok in file_audit():
            tag = "OK  " if ok else "MISS"
            colour = "#2a9d8f" if ok else "#c1121f"
            st.markdown(f"<span style='color:{colour};font-weight:700'>[{tag}]</span> "
                        f"<code>{path}</code> — {label}", unsafe_allow_html=True)

    tab_map, tab_vessel, tab_cal, tab_cmp, tab_method = st.tabs(
        ["Map", "Vessel drill-down", "Calibration", "Compare", "Methodology"])

    # ---------------- 1 + 2 + 4 + 5: map ----------------
    with tab_map:
        # ---- 4: contextual info card on arrival at a fly-to location ----
        arr = st.session_state.get("arrived")
        if arr and arr in LOCATIONS:
            loc = LOCATIONS[arr]
            st.markdown(
                "<div style='background:linear-gradient(135deg,#0b0f14,#122733);"
                "border:1px solid #2b6b7d;border-left:4px solid #4cc9f0;border-radius:8px;"
                "padding:12px 16px;margin:2px 0 12px;max-width:780px'>"
                "<div style='font-size:10px;letter-spacing:1.5px;color:#6b8ea0;"
                "text-transform:uppercase'>Location brief</div>"
                f"<div style='font-size:15px;font-weight:700;color:#e6edf3;margin-top:2px'>{loc['label']}</div>"
                f"<div style='font-size:12px;color:#9fb3c8;margin-top:4px'>{loc['context']}</div>"
                f"<div style='font-size:12.5px;color:#e9c46a;margin-top:7px'>"
                f"<b>Project finding:</b> {loc['finding']}</div></div>",
                unsafe_allow_html=True)

        st.subheader(wname)
        df = _window(wname)
        if df is None or df.empty:
            st.error(f"Could not load window file for **{wname}**.")
        else:
            kind = WINDOWS[wname]["kind"]
            counts = df["cls"].value_counts()
            cols = st.columns(4)
            for c, cc in zip(cols, CANON):
                c.metric(cc, int(counts.get(cc, 0)),
                         f"{100 * counts.get(cc, 0) / len(df):.0f}%")
            if kind == "relinked":
                st.info("India is the **SAR↔AIS re-linking layer**, not a spoofing "
                        "classification. Buckets map link outcomes: gfw_locked→matched, "
                        "relinked→discrepant, forced→likely_spoofed, dark_vessel→unmatched.")

            # ---- view state: fly-to target, or this window's centre ----
            if st.session_state.get("view") is None:
                st.session_state.view = dict(latitude=float(df["lat"].mean()),
                                             longitude=float(df["lon"].mean()), zoom=5.6)
            view = st.session_state.view

            loiter_label = WINDOW_LOITER_LABEL.get(wname)
            dfl = _loiter(loiter_label)
            tracks = _tracks(RAW_AIS_LABEL.get(wname)) if st.session_state.get("draw_tracks", True) else {}
            scores_mmsi = (set(int(x) for x in scores["mmsi"].dropna())
                           if scores is not None else set())
            loiter_mmsi = (set(int(x) for x in dfl["mmsi"].dropna())
                           if dfl is not None and not dfl.empty else set())

            # resolve a prior click (widget state) to a MMSI for the full-track overlay
            _selstate = st.session_state.get("mainmap") or {}
            try:
                _hit = ((_selstate.get("selection") or {}).get("objects", {}) or {}).get("ships") or []
                picked_mmsi = _hit[0].get("mmsi_i") if _hit else None
            except Exception:
                picked_mmsi = None

            flagged_only = not st.session_state.get("show_all", False)
            rec = build_map_records(
                df, tracks, kind, scores_mmsi, loiter_mmsi,
                trail_hours=st.session_state.get("trail_hours", 18),
                flagged_only=flagged_only,
                show_unmatched=st.session_state.get("show_bg", False),
                atlas=_atlas())

            rmode = st.session_state.get("render_mode", "Auto (cluster when zoomed out)")
            if rmode == "Individual ships":
                detail = "individual"
            elif rmode == "Clustered density":
                detail = "clustered"
            else:
                detail = "clustered" if view["zoom"] < 8.3 else "individual"

            facs = [f for _lc in LOCATIONS.values() if _lc["window"] == wname
                    for f in _lc["facilities"]]
            deck = build_map_deck(rec, facility_records(facs), loiter_records(dfl),
                                  view, detail=detail, picked_mmsi=picked_mmsi)

            _, _, basemap_label, _ = deck_basemap()
            st.markdown(_map_legend_html(kind, st.session_state.get("show_bg", False)),
                        unsafe_allow_html=True)
            st.caption(f"Trail opacity fades oldest → newest, capped to the last "
                       f"{st.session_state.get('trail_hours', 18)} h · "
                       f"{'density clustering (low zoom)' if detail == 'clustered' else 'individual ships'} · "
                       f"basemap {basemap_label}")

            event = st.pydeck_chart(
                deck, width="stretch", height=600,
                selection_mode="single-object", on_select="rerun", key="mainmap")
            _bg = ", + background traffic" if st.session_state.get("show_bg") else ""
            st.caption(
                f"{len(rec['icons'])} of {len(df)} SAR detections shown "
                f"({'flagged & escalated only' if flagged_only else 'all classified'}{_bg}) · "
                f"{len(rec['recent_dots'])} with a capped trail · "
                f"{len(rec.get('conn_segs', []))} SAR↔AIS connector line(s). "
                "Use the sidebar **Fly to** list to move between locations; hover a ship for a "
                "readout, click it for the full panel and its complete track.")

            # ---------------- click-to-inspect (same detail panel) ------
            picked = None
            try:
                objs = (event.selection or {}).get("objects", {}) or {}
                hit = objs.get("ships") or []
                picked = hit[0] if hit else None
            except Exception:
                picked = None
            if picked:
                st.divider()
                mm = picked.get("mmsi_i")
                srow = None
                if scores is not None and mm is not None:
                    m_ = scores[scores["mmsi"] == mm]
                    srow = m_.iloc[0] if len(m_) else None
                st.markdown(f"### {picked.get('name', '?')}  ·  MMSI {picked.get('mmsi', '—')}")
                st.write(f"**Classification:** {picked.get('cls')}  ·  "
                         f"{picked.get('val_label')}: {picked.get('val')}  ·  "
                         "_full track drawn on the map_")
                if srow is not None:
                    render_vessel_detail(st, scores, srow, _deep)
                else:
                    st.caption("Not in the v6 flagged leaderboard — no reliability score "
                               "or deep-history for this vessel. Use the **Vessel drill-down** "
                               "tab to search the scored fleet.")

            # ---------------- 5: loiter / dwell detector for this window ----
            if dfl is not None and not dfl.empty:
                with st.expander(
                        f"Loiter detector — {len(dfl)} prolonged-dwell escalation(s) "
                        "in this window", expanded=("Jamnagar" in wname)):
                    st.caption(
                        "Vessels that sat still ≥ 7 days within 25 km of an EU/UK-sanctioned "
                        "facility — a structurally different flag (stillness, not movement) that is "
                        "invisible to the trajectory and cluster stages. Drawn as magenta haloed "
                        f"markers on the map. Source: `loiter_flags_{loiter_label}.csv`.")
                    _show = dfl[["ship_name", "mmsi", "flag", "duration_h", "radius_km",
                                 "nearest_facility", "nearest_facility_km",
                                 "episode_start", "episode_end"]].copy()
                    _show.insert(4, "days", (_show["duration_h"] / 24).round(1))
                    st.dataframe(_show, hide_index=True, width="stretch")

            # ---------------- 2: GFW link vs corrected re-link (India only) --
            #        both drawn at once for a direct before/after comparison
            if kind == "relinked":
                st.divider()
                st.markdown("#### GFW's original link vs. the corrected Hungarian re-link "
                            "— shown together")
                pairs = _india_pairs()
                if pairs is None or pairs.empty:
                    st.info("No `gfw_corrected` rows in india_mangalore_relinked.csv.")
                else:
                    n_tail = int(pairs["tail_outlier"].sum())
                    st.caption(
                        f"{len(pairs)} SAR detections where the Hungarian re-linker overrode "
                        "GFW's MMSI. GFW's original link is drawn thin / dashed / muted-gray; the "
                        "corrected re-link is solid, glowing and colour-coded with a diamond "
                        f"endpoint. Tukey far-out fence ({INDIA_TAIL_FENCE_KM:g} km): {n_tail} of "
                        f"{len(pairs)} pairs clear it (the ones inspect_india_tail_outliers.py isolated).")
                    only_tail = st.checkbox("Only fence-clearing (tail-outlier) pairs",
                                            value=False, key="relink_tail")
                    pp = pairs[pairs["tail_outlier"]] if only_tail else pairs
                    if pp.empty:
                        pp = pairs
                    lbls = [f"SAR {int(x.sar_id)} · GFW {x.gfw_name} ({x.gfw_mmsi}) → "
                            f"re-link {x.relinked_name} ({x.relinked_mmsi})"
                            for x in pp.itertuples()]
                    sel = st.selectbox("Pair", lbls, key="relink_pair")
                    pr = pp.iloc[lbls.index(sel)]

                    gfw_ok = bool(pd.notna(pr["gfw_lat"]))
                    sar_xy = [float(pr["sar_lon"]), float(pr["sar_lat"])]
                    rel_xy = [float(pr["rel_lon"]), float(pr["rel_lat"])]
                    REL_RGB = [42, 157, 143]
                    layers = []

                    # GFW's original link: thin, dashed, muted
                    if gfw_ok:
                        gfw_xy = [float(pr["gfw_lon"]), float(pr["gfw_lat"])]
                        layers.append(pdk.Layer(
                            "PathLayer", data=_dash_segments(sar_xy, gfw_xy), id="gfw_dash",
                            get_path="path", get_color=[150, 150, 158, 205],
                            get_width=1.6, width_min_pixels=1.4, pickable=False))
                        layers.append(pdk.Layer(
                            "ScatterplotLayer",
                            data=[{"position": gfw_xy, "name": f"GFW: {pr['gfw_name']}",
                                   "mmsi": str(pr["gfw_mmsi"]), "cls": "GFW original link",
                                   "val_label": "reconstructed separation",
                                   "val": f"≈ {pr['gfw_dist_km']:.1f} km"}],
                            id="gfw_pt", get_position="position",
                            get_fill_color=[150, 150, 158, 215], stroked=True,
                            get_line_color=[222, 222, 226, 230], line_width_min_pixels=1,
                            get_radius=6, radius_min_pixels=5, radius_max_pixels=9,
                            pickable=True))

                    # corrected re-link: glow + solid core + glowing diamond endpoint
                    seg = [{"path": [sar_xy, rel_xy]}]
                    layers.append(pdk.Layer("PathLayer", data=seg, id="relink_glow",
                                            get_path="path", get_color=REL_RGB + [55],
                                            get_width=9, width_min_pixels=8,
                                            cap_rounded=True, joint_rounded=True))
                    layers.append(pdk.Layer("PathLayer", data=seg, id="relink_core",
                                            get_path="path", get_color=REL_RGB + [255],
                                            get_width=3, width_min_pixels=2.4,
                                            cap_rounded=True, joint_rounded=True))
                    for rad, a in ((16, 70), (9, 255)):
                        layers.append(pdk.Layer(
                            "ScatterplotLayer",
                            data=[{"position": rel_xy,
                                   "name": f"re-link: {pr['relinked_name']}",
                                   "mmsi": str(pr["relinked_mmsi"]),
                                   "cls": "corrected re-link", "val_label": "re-link distance",
                                   "val": f"{pr['relinked_dist_km']:.2f} km"}],
                            id=f"relink_end_{rad}", get_position="position",
                            get_fill_color=REL_RGB + [a], get_radius=rad,
                            radius_min_pixels=rad, radius_max_pixels=rad + 4,
                            stroked=(rad == 9), get_line_color=[240, 255, 250, 235],
                            line_width_min_pixels=1, pickable=(rad == 9)))
                    layers.append(pdk.Layer(
                        "ScatterplotLayer",
                        data=[{"position": sar_xy, "name": f"SAR detection {int(pr['sar_id'])}",
                               "mmsi": "—", "cls": "SAR detection",
                               "val_label": "timestamp", "val": str(pr["sar_ts"])}],
                        id="sar_pt", get_position="position",
                        get_fill_color=[236, 236, 236, 235], stroked=True,
                        get_line_color=[20, 20, 20, 255], line_width_min_pixels=2,
                        get_radius=7, radius_min_pixels=7, radius_max_pixels=10, pickable=True))

                    _, style2, _, tok2 = deck_basemap()
                    provider2 = "mapbox" if tok2 else "carto"
                    relink_deck = pdk.Deck(
                        layers=layers,
                        initial_view_state=pdk.ViewState(
                            latitude=float(pr["sar_lat"]), longitude=float(pr["sar_lon"]),
                            zoom=9.5, pitch=0, bearing=0),
                        map_provider=provider2, map_style=style2,
                        api_keys=({"mapbox": tok2} if tok2 else None),
                        tooltip={"html": "<b>{name}</b><br/>MMSI: {mmsi}<br/>{cls}"
                                         "<br/>{val_label}: {val}",
                                 "style": {"backgroundColor": "#0b0f14",
                                           "color": "#e6edf3", "fontSize": "12px"}})
                    st.markdown(
                        "<div style='font-size:12.5px'>"
                        "<span style='color:#9696a0'>╌╌</span> GFW original link "
                        "(thin, dashed, muted) &nbsp;·&nbsp; "
                        "<span style='color:#2a9d8f'>━◆</span> corrected re-link "
                        "(solid, glowing, diamond endpoint) &nbsp;·&nbsp; "
                        "<span style='color:#ececec'>●</span> SAR detection</div>",
                        unsafe_allow_html=True)
                    st.pydeck_chart(relink_deck, width="stretch", height=440,
                                    key=f"relink_pydeck_{int(pr['sar_id'])}")
                    if not gfw_ok:
                        st.warning("GFW's MMSI has no cached AIS fix near this SAR time — "
                                   "its dashed line can't be drawn.")
                    mc = st.columns(2)
                    mc[0].metric("GFW original separation",
                                 "n/a" if pd.isna(pr["gfw_dist_km"]) else f"{pr['gfw_dist_km']:.1f} km")
                    mc[1].metric("Corrected re-link distance",
                                 f"{pr['relinked_dist_km']:.2f} km")
                    st.caption("SAR point + corrected re-link distance: read from "
                               "india_mangalore_relinked.csv. GFW's original vessel position: "
                               "nearest-time lookup in the cached raw India AIS file "
                               "(that separation is not persisted anywhere).")

    # ---------------- 3: vessel drill-down ----------------
    with tab_vessel:
        st.subheader("Vessel drill-down")
        if scores is None:
            st.error(f"`{SCORES_FILE}` not found — drill-down unavailable.")
        else:
            q = st.text_input("Search by vessel name or MMSI", "SELENIA").strip()
            hits = scores
            if q:
                mask = scores["ship_name"].fillna("").str.contains(q, case=False)
                if q.isdigit():
                    mask = mask | scores["mmsi"].astype(str).str.contains(q)
                hits = scores[mask]
            if len(hits) == 0:
                st.warning("No vessel in the v6 leaderboard matches that search.")
            else:
                labels = hits.apply(
                    lambda r: f"{r['ship_name']}  ({int(r['mmsi'])})  — score {r['reliability_score']:g}",
                    axis=1).tolist()
                choice = st.selectbox("Match", labels)
                row = hits.iloc[labels.index(choice)]
                render_vessel_detail(st, scores, row, _deep)

    # ---------------- 3b: threshold calibration histogram ----------------
    with tab_cal:
        st.subheader("Threshold calibration — confirmed SAR↔AIS pair distances")
        st.caption("Distance distribution of confirmed same-vessel (identity-matched) "
                   "pairs, with the derived p50 (matched) / p95 / p99 (likely_spoofed) "
                   "cutoffs. Source: `threshold_calibration_{hormuz_control,india_mangalore}.json` "
                   "(calibrate_thresholds.py).")

        def _hist_panel(container, title, dists, cal, color, real):
            with container:
                st.markdown(f"**{title}**")
                if cal is None or dists is None:
                    st.warning("calibration data missing.")
                    return
                fig, ax = plt.subplots(figsize=(5.2, 3.6))
                ax.hist(dists, bins=24, color=color, alpha=0.82,
                        edgecolor="white", linewidth=0.4)
                ymax = ax.get_ylim()[1]
                for x, lab, lc in [
                        (cal["matched_km"], "p50\nmatched", "#2a9d8f"),
                        (cal["p95_km"], "p95", "#e9a02c"),
                        (cal["likely_spoofed_km"], "p99\nlikely_spoofed", "#c1121f")]:
                    ax.axvline(x, color=lc, ls="--", lw=1.6)
                    ax.text(x, ymax * 0.98, f" {lab}\n {x:.1f} km", fontsize=7,
                            color=lc, va="top", ha="left")
                ax.set_xlabel("nearest-AIS distance (km)")
                ax.set_ylabel("confirmed pairs")
                ax.set_title(f"n≈{len(dists)}  ·  confidence: {cal['confidence']}",
                             fontsize=9)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                if real:
                    st.caption(f"REAL per-pair distances — the {cal['n_pre_screen']} pre-screen "
                               "confirmed identity pairs in `hormuz_control_mar2026_classified.csv` "
                               f"(`distance_km`). Cutoffs derived on the {cal['n_confirmed']} "
                               f"post-screen pairs ({cal['n_excluded']} misattributed pairs removed).")
                else:
                    st.caption("RECONSTRUCTED from the stored percentile grid "
                               "(`confirmed_pair_percentiles_km`, p10…p99.9) by inverse-CDF "
                               "interpolation — calibrate_thresholds.py does not persist India's "
                               "raw per-pair distances. The dashed cutoff lines are the real "
                               f"derived values ({cal['n_confirmed']} post-screen pairs, "
                               f"{cal['n_excluded']} removed).")

        gcal, ical = _calib("gulf"), _calib("india")
        pc = st.columns(2)
        _hist_panel(pc[0], "Gulf — Hormuz control (Jan 2026)",
                    _gulf_dists(), gcal, "#264653", real=True)
        _hist_panel(pc[1], "India — New Mangalore (Sep–Oct 2025)",
                    (reconstruct_dist_from_percentiles(ical["confirmed_pct"])
                     if ical else None), ical, "#7a5195", real=False)

        if gcal and ical:
            st.markdown(
                f"- **Gulf** p50 / p95 / p99 = {gcal['matched_km']:.2f} / "
                f"{gcal['p95_km']:.1f} / {gcal['likely_spoofed_km']:.2f} km  \n"
                f"- **India** p50 / p95 / p99 = {ical['matched_km']:.2f} / "
                f"{ical['p95_km']:.1f} / {ical['likely_spoofed_km']:.2f} km  \n"
                "Both flagged **low-confidence**: a single short control window, "
                "< 200 confirmed pairs, p99 near the sample maximum.")

    # ---------------- 4: compare ----------------
    with tab_cmp:
        st.subheader("Comparison")
        reg = WINDOW_REGION.get(wname)
        reg_name = REGION_LABEL.get(reg, str(reg))

        # ---- Section A: same-region classification comparison (valid) ----
        st.markdown(f"#### Same-region classification comparison — {reg_name}")
        st.caption(
            "Valid only within one region: same waters, same independently-calibrated "
            "threshold set, an incident window against its own control. Same-region "
            "pairs: Qatar incident vs. Qatar control · Hormuz crisis vs. Hormuz control · "
            "New Mangalore vs. Jamnagar/Vadinar.")
        if not wcompare or wcompare == wname:
            st.info(f"No second **{reg_name}** window is available to compare against.")
        else:
            da, db = _window(wname), _window(wcompare)
            if da is None or db is None:
                st.error("One of the two window files could not be loaded.")
            else:
                st.markdown(f"**{wname}**  vs  **{wcompare}**")
                if WINDOWS[wname]["kind"] != WINDOWS[wcompare]["kind"]:
                    st.warning(
                        "These two windows are different output layers — spoofing "
                        "classification vs. SAR↔AIS re-linking. The bucket names line up "
                        "but the underlying measurement does not; read each set of bars "
                        "as a within-window mix, not a like-for-like difference.")
                pa, pb = _cls_pct(da), _cls_pct(db)
                fig, ax = plt.subplots(figsize=(8, 3.4))
                x = np.arange(len(CANON))
                ax.bar(x - 0.2, [pa[c] for c in CANON], 0.4, label=wname, color="#264653")
                ax.bar(x + 0.2, [pb[c] for c in CANON], 0.4, label=wcompare, color="#e76f51")
                ax.set_xticks(x)
                ax.set_xticklabels(CANON)
                ax.set_ylabel("% of SAR detections")
                ax.legend(fontsize=8)
                for i, c in enumerate(CANON):
                    ax.text(i - 0.2, pa[c] + 1, f"{pa[c]:.0f}", ha="center", fontsize=8)
                    ax.text(i + 0.2, pb[c] + 1, f"{pb[c]:.0f}", ha="center", fontsize=8)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        # ---- Section B: cross-region THRESHOLD comparison (values only) ----
        st.divider()
        st.markdown("#### Cross-region threshold / methodology comparison")
        st.caption(
            "Qatar, Hormuz and India are calibrated **independently** — each has its own "
            "confirmed-pair distance distribution and its own derived p50 / p95 / p99 "
            "cutoffs. Only the **threshold values themselves** are comparable across "
            "regions. Raw classification percentages are NOT a valid cross-region "
            "activity-level comparison (different waters, traffic mix, AIS coverage and "
            "cutoffs). Histograms behind these numbers: Calibration tab.")
        gcal, ical = _calib("gulf"), _calib("india")
        rows = []
        for cal, rlabel in ((gcal, "Strait of Hormuz (Gulf calibration)"),
                            (ical, "India west coast (New Mangalore calibration)")):
            if cal:
                rows.append({
                    "region": rlabel,
                    "p50 matched (km)": round(float(cal["matched_km"]), 2),
                    "p95 (km)": round(float(cal["p95_km"]), 1),
                    "p99 likely_spoofed (km)": round(float(cal["likely_spoofed_km"]), 2),
                    "confirmed pairs": int(cal["n_confirmed"]),
                    "confidence": cal["confidence"],
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.warning("No calibration JSON available.")
        st.markdown("##### SAR↔AIS re-linking: derived `likely_spoofed` threshold by method (India)")
        p = f"{FIGS}/threshold_correction_comparison.png"
        if os.path.exists(p):
            st.image(p)
        else:
            st.warning(f"`{p}` not found.")

    # ---------------- 5: methodology ----------------
    with tab_method:
        _render_methodology(st, scores)


def _rank(scores, mmsi):
    nonart = scores[~scores["is_known_artifact"]].sort_values(
        ["reliability_score", "flag_rate", "flagged_event_count"], ascending=False
    ).reset_index(drop=True)
    hit = nonart.index[nonart["mmsi"] == mmsi]
    return f"#{int(hit[0]) + 1} / {len(nonart)}" if len(hit) else "n/a"


def _cls_pct(df):
    vc = df["cls"].value_counts()
    n = max(len(df), 1)
    return {c: 100 * vc.get(c, 0) / n for c in CANON}


def _render_deep(st, dh):
    pv = dh[dh["event_type"] == "port_visit"]
    lo = dh[dh["event_type"] == "loitering"]
    gap = dh[dh["event_type"] == "ais_gap"]
    countries = sorted(set(dh["eez"].dropna()) | set(dh["port_flag"].dropna()))
    span = f"{dh['start'].min():%Y-%m-%d} → {dh['end'].max():%Y-%m-%d}"
    max_lo = float(lo["duration_hrs"].max()) if len(lo) else 0.0
    st.markdown("#### 6-month operating summary (deep-history)")
    c = st.columns(4)
    c[0].metric("port calls", len(pv))
    c[1].metric("loitering events", len(lo))
    c[2].metric("AIS gaps", len(gap))
    c[3].metric("longest loiter", f"{max_lo:.0f} h")
    st.write(f"**Window:** {span}  ·  **EEZ / port countries:** "
             + ", ".join(countries[:12]) + ("…" if len(countries) > 12 else ""))
    ports = sorted(pv["port_name"].dropna().unique())
    if ports:
        st.write("**Ports visited:** " + ", ".join(ports[:20]))
    long_lo = lo[lo["duration_hrs"] > 24]
    if len(long_lo):
        st.write(f"**Multi-day loitering:** {len(long_lo)} episode(s) > 24 h "
                 f"(max {max_lo:.0f} h) — see deep-history CSV for anchorage locations.")


def _render_methodology(st, scores):
    st.subheader("Methodology & context (static — for live demo)")

    st.markdown("### 6 real bugs found & fixed")
    for i, (title, fname, line) in enumerate(BUGS, 1):
        st.markdown(f"**{i}. {title}**  — `{fname}`  ·  {line}")
    p = f"{FIGS}/bugs_timeline.png"
    if os.path.exists(p):
        with st.expander("bugs timeline figure"):
            st.image(p)

    st.divider()
    st.markdown("### The top-15 leaderboard is fully externally verified (v6)")
    st.write("Every vessel in the current top-15 has been checked — either an external "
             "registry confirmation (SELENIA, HAKKAISAN, PATRIS, OCEAN CENTURY, LENORE) or "
             "an unambiguous GFW-native type. Two corrections are the concrete before/after "
             "(v4 = age component just fixed; v5 = tug down-weight added; v6 = HAKKAISAN age confirmed):")
    priors = {p: (pd.read_csv(f"{PROC}/vessel_reliability_scores_{p}.csv")
                  if os.path.exists(f"{PROC}/vessel_reliability_scores_{p}.csv") else None)
              for p in ("v4", "v5")}
    if scores is not None and all(v is not None for v in priors.values()):
        rows = []
        for mmsi, before_ver, note in [
                (636025162, "v4", "GFW typed it NA → v2 tug down-weight was missed; the age fix then lifted it "
                                  "to #2. Registry-confirmed tug → behavioural ×0.25 applied."),
                (352003690, "v5", "GFW typed it OTHER, no IMO → age never resolved. Registry-confirmed genuine "
                                  "VLCC → age>15 yr +20, NO down-weight.")]:
            bdf = priors[before_ver]
            a = bdf[bdf["mmsi"] == mmsi].iloc[0]
            b = scores[scores["mmsi"] == mmsi].iloc[0]
            rows.append({"vessel": b["ship_name"],
                         "before": f"{a['reliability_score']:g}  ({_rank(bdf, mmsi)}, {before_ver})",
                         "after (v6, final)": f"{b['reliability_score']:g}  ({_rank(scores, mmsi)})",
                         "correction": note})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.caption("(prior-version files not found — before/after table skipped.)")
    p = f"{FIGS}/vessel_score_breakdown.png"
    if os.path.exists(p):
        with st.expander("score-breakdown figure (v6)"):
            st.image(p)

    st.divider()
    st.markdown("### LENORE — the 4-part case")
    st.info(
        "**Identity** — 3-flag reflagging history, shadow-fleet flag (BES), FOC +20.  \n"
        "**Physics** — a single 1-hour AIS step of ~152 km → **82 kn implied speed**, "
        "physically impossible for a tanker.  \n"
        "**Location** — the jump lands inside the Hormuz crisis window (Mar 2026), "
        "in the strait, exactly when interference was reported.  \n"
        "**Own-history** — its own hourly AIS shows the position **frozen for ~4 days** "
        "(≤0.54 kn) immediately before the jump, then a clean decay back to 7–13 kn "
        "transit — the classic freeze-then-teleport spoof signature, visible without any "
        "external reference.")
    p = f"{FIGS}/two_tier_architecture.png"
    if os.path.exists(p):
        with st.expander("real-time architecture (Tier 1 screening → Tier 2 investigation)"):
            st.image(p)


# ==========================================================================
def _under_streamlit():
    """True when launched via `streamlit run`, False for a plain `python`."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def _print_audit():
    print("Data-file audit for src/dashboard.py")
    print("=" * 70)
    ok = bad = 0
    for label, path, exists in file_audit():
        print(f"  {'OK  ' if exists else 'MISS'}  {path}\n        {label}")
        ok += exists
        bad += (not exists)
    print("=" * 70)
    print(f"  {ok} present, {bad} missing")
    print("\nRun the dashboard with:\n  streamlit run src/dashboard.py")


if _under_streamlit():
    run_app()
elif __name__ == "__main__":
    _print_audit()
