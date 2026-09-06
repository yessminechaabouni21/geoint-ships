"""Vessel-specific baseline comparison for a small shortlist of vessels.

WHAT THIS IS (and how it differs from src/vessel_history.py)
-----------------------------------------------------------
src/vessel_history.py builds a *cross-incident-window consistency* score: it
asks "how often was this vessel flagged across the 4 sparse, incident-anchored
5-day windows we happen to have SAR/AIS for?" That is a comparison of a vessel
against OUR sampling, not against the vessel's own normal behaviour. A vessel
that was flagged in 1 of 1 observed windows scores a high flag_rate purely
because the sample is tiny.

This module does the opposite: it pulls each shortlisted vessel's own
CONTINUOUS event history from GFW over a long, unbroken period (Sept 1 2025 -
Mar 24 2026, i.e. spanning every incident window) and establishes that
vessel's *individual baseline*: where it normally operates, how it normally
moves, how it normally reports AIS, where it calls. It then asks whether the
behaviour we flagged in the incident windows sits INSIDE or OUTSIDE that
vessel's own baseline. The output concept is therefore
"vessel-specific baseline comparison", NOT "historical reliability".

CRITICAL - the circularity trap (see cross_reference() and the printed
summary). If a flagged behaviour turns out to MATCH the vessel's 6-month
baseline, that is NOT automatically "nothing unusual, cleared". Two readings
are always possible and BOTH are printed:
  (a) the behaviour is genuinely ordinary for this vessel - a real
      false positive of the sparse-window method; or
  (b) the suspicious-looking pattern is PERSISTENT across the whole 6 months
      rather than a one-off - which, for a vessel that also carries
      independent identity red flags (shadow-fleet flag, reflagging history,
      sanctioned-adjacent routing, long lay-ups, recent renaming), is a
      STRONGER finding, not a weaker one.
The module scores explicit baseline-anomaly indicators and states which
reading the evidence better supports for each vessel, rather than defaulting
to the reassuring one.

GFW API notes (confirmed against the live v3 API this run - see probe output)
--------------------------------------------------------------------------
* There is NO per-vessel position track endpoint. /vessels/{id}/tracks,
  /vessels/{id}/track and /4wings/track all 404 (also documented in
  src/config.py, verified 2026-09-03; re-confirmed here).
* The per-vessel time-series API that DOES exist is /events, queried with
  vessels[0]={vesselId}. It returns discrete events (port visits, AIS gaps,
  encounters, loitering, fishing) with precise sub-minute timestamps and
  exact lat/lon - FINER resolution than the hourly ~1/100-degree regional
  4wings presence grid, but sparse (event-anchored, not continuous track).
* Consequence: "typical speed" cannot be read off a track. It is
  reconstructed here two coarser ways, both labelled as estimates:
    - at-rest speed distribution from loitering events' averageSpeedKnots
    - implied transit speed = great-circle distance between consecutive port
      calls / elapsed time between them.

Run:  python -m src.vessel_deep_history [--no-gfw] [--refresh-gfw]
Outputs: data/processed/vessel_deep_history_{mmsi}.csv   (one file per vessel)
         data/processed/gfw_deep_events_cache.json        (raw event cache)
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import GFW_API_BASE_URL, GFW_API_TOKEN

PROC = "data/processed"
DEEP_CACHE_PATH = f"{PROC}/gfw_deep_events_cache.json"
V3_SCORES = f"{PROC}/vessel_reliability_scores_v3.csv"

# Continuous baseline period -- unbroken, and wide enough to contain every
# incident window (Qatar Sep/Oct 2025, Hormuz Jan 2026, Hormuz Mar 2026).
BASELINE_START = "2025-09-01"
BASELINE_END = "2026-03-24"

SHORTLIST = {
    511101414: "SELENIA",
    636018010: "PATRIS",
    636025162: "OCEAN CENTURY",
}

EVENT_DATASETS = {
    "port_visit": "public-global-port-visits-events:latest",
    "gap": "public-global-gaps-events:latest",
    "encounter": "public-global-encounters-events:latest",
    "loitering": "public-global-loitering-events:latest",
    "fishing": "public-global-fishing-events:latest",
}

# EEZ sovereign-code -> label. GFW `regions.eez` carries Marine Regions MRGID
# strings; only the ones this shortlist actually touches are named, everything
# else falls through to the raw code (printed, never silently dropped).
EEZ_LABEL = {
    "8480": "India", "8360": "United Arab Emirates", "8470": "Iraq",
    "8468": "Qatar", "8469": "Iran", "8358": "Bahrain", "8356": "Kuwait",
    "8354": "Oman", "8487": "Japan", "8492": "Indonesia", "8483": "Malaysia",
    "8327": "South Korea", "8332": "Thailand", "8323": "Australia",
    "8345": "Maldives", "8346": "Sri Lanka",
    "48947": "Iran-UAE overlapping claim", "8471": "Saudi Arabia",
}

# --- India-area test (bonus check, requirement 5) -------------------------
# Any of: GFW anchorage id prefixed "ind-", startAnchorage.flag == "IND",
# EEZ code 8480 (India) in an event's regions, or an event position inside
# this generous Indian-waters box. Named ports of interest are listed so a
# match can be reported by name.
INDIA_EEZ_CODE = "8480"
INDIA_BBOX = (6.0, 24.5, 68.0, 90.0)  # (min_lat, max_lat, min_lon, max_lon)
INDIA_PORTS = {
    "jamnagar": (22.47, 69.08), "vadinar": (22.28, 69.73), "sikka": (22.43, 69.84),
    "kandla": (23.03, 70.22), "mundra": (22.74, 69.70), "mumbai": (18.95, 72.83),
    "jnpt": (18.95, 72.95), "nhava_sheva": (18.95, 72.95), "paradip": (20.26, 86.67),
    "new_mangalore": (12.93, 74.81), "mangalore": (12.93, 74.81),
    "cochin": (9.97, 76.24), "chennai": (13.10, 80.30), "sikka_jetty": (22.43, 69.84),
    "vizag": (17.69, 83.30), "visakhapatnam": (17.69, 83.30),
}


# ==========================================================================
# GFW access
# ==========================================================================
def _session():
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {GFW_API_TOKEN}"})
    return s


def _get(sess, path, **params):
    """GET with light retry on transient failure / rate limit."""
    url = f"{GFW_API_BASE_URL}{path}"
    for attempt in range(4):
        try:
            r = sess.get(url, params=params, timeout=90)
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 429:
            time.sleep(3 * (attempt + 1))
            continue
        return r
    return r


def resolve_vessel(sess, mmsi):
    """MMSI -> best current GFW identity. Returns dict with vessel_id plus the
    identity facts that matter for the baseline-anomaly scorecard (current
    flag, imo, and the full reflagging / renaming history)."""
    r = _get(sess, "/vessels/search", query=str(mmsi), limit=10,
             **{"datasets[0]": "public-global-vessel-identity:latest",
                "includes[0]": "OWNERSHIP"})
    r.raise_for_status()
    entries = r.json().get("entries") or []
    srp = []
    for e in entries:
        srp += (e.get("selfReportedInfo") or [])
    # identities that actually reported as THIS mmsi
    mine = [s for s in srp if str(s.get("ssvid")) == str(mmsi)]
    if not mine:
        return {"mmsi": mmsi, "vessel_id": None, "identity_available": False}
    # "current" = latest transmissionDateTo, and prefer one that has a name
    def _key(s):
        return (s.get("shipname") is not None, s.get("transmissionDateTo") or "")
    cur = sorted(mine, key=_key)[-1]

    # full history (all ssvid/flag/name the same hull has used), for the
    # reflagging-history anomaly indicator
    hist = []
    for s in sorted(srp, key=lambda x: x.get("transmissionDateFrom") or ""):
        # skip low-count identity fragments (GFW sometimes emits a stub
        # self-reported record with a few hundred positions and no name/imo)
        if (s.get("positionsCounter") or 0) < 1000 and not s.get("shipname"):
            continue
        hist.append({
            "ssvid": s.get("ssvid"), "flag": s.get("flag"),
            "name": s.get("shipname"), "imo": s.get("imo"),
            "from": (s.get("transmissionDateFrom") or "")[:10],
            "to": (s.get("transmissionDateTo") or "")[:10],
            "positions": s.get("positionsCounter"),
        })
    return {
        "mmsi": mmsi,
        "vessel_id": cur.get("id"),
        "identity_available": True,
        "name": cur.get("shipname"),
        "flag": cur.get("flag"),
        "imo": cur.get("imo"),
        "positions_counter": cur.get("positionsCounter"),
        "messages_counter": cur.get("messagesCounter"),
        "transmission_from": (cur.get("transmissionDateFrom") or "")[:10],
        "transmission_to": (cur.get("transmissionDateTo") or "")[:10],
        "current_name_since": (cur.get("transmissionDateFrom") or "")[:10],
        "identity_history": hist,
        "n_distinct_flags": len({h["flag"] for h in hist if h["flag"]}),
        "n_distinct_names": len({h["name"] for h in hist if h["name"]}),
    }


def probe_track_endpoint(sess, vessel_id):
    """Requirement 1: confirm the per-vessel track endpoint rather than
    assuming. Returns a short status string; expected to be 404."""
    try:
        r = _get(sess, f"/vessels/{vessel_id}/tracks",
                 **{"start-date": BASELINE_START, "end-date": "2025-09-05",
                    "datasets[0]": "public-global-fishing-tracks:latest"})
        return f"HTTP {r.status_code} ({'no per-vessel track' if r.status_code == 404 else 'unexpected - inspect'})"
    except Exception as exc:  # noqa: BLE001
        return f"error: {type(exc).__name__}"


def fetch_events(sess, vessel_id, kind, dataset):
    rows, offset = [], 0
    while True:
        r = _get(sess, "/events", **{
            "vessels[0]": vessel_id, "datasets[0]": dataset,
            "start-date": BASELINE_START, "end-date": BASELINE_END,
            "limit": 100, "offset": offset,
        })
        if r.status_code != 200:
            print(f"    ! {kind}: HTTP {r.status_code} {r.text[:120]}")
            break
        j = r.json()
        page = j.get("entries") or []
        rows += page
        nxt = j.get("nextOffset")
        if nxt is None or not page:
            break
        offset = nxt
        time.sleep(0.1)
    return rows


def load_all_events(enabled=True, refresh=False):
    cache = {}
    if Path(DEEP_CACHE_PATH).exists() and not refresh:
        try:
            cache = json.loads(Path(DEEP_CACHE_PATH).read_text())
        except json.JSONDecodeError:
            cache = {}

    need = [m for m in SHORTLIST if refresh or str(m) not in cache]
    if need and not enabled:
        print("  --no-gfw and cache incomplete: cannot build baseline for "
              f"{need}. Re-run without --no-gfw once.")
    if need and enabled:
        if not GFW_API_TOKEN:
            raise RuntimeError("GFW_API_TOKEN not set - cannot pull deep history.")
        sess = _session()
        for mmsi in need:
            print(f"  {SHORTLIST[mmsi]} ({mmsi}): resolving identity ...")
            ident = resolve_vessel(sess, mmsi)
            vid = ident.get("vessel_id")
            if not vid:
                print(f"    ! no GFW vessel id for {mmsi}; skipping")
                cache[str(mmsi)] = {"identity": ident, "events": {}, "track_probe": None}
                continue
            track_probe = probe_track_endpoint(sess, vid)
            print(f"    vessel_id={vid}  track endpoint: {track_probe}")
            events = {}
            for kind, ds in EVENT_DATASETS.items():
                ev = fetch_events(sess, vid, kind, ds)
                events[kind] = ev
                print(f"    {kind:10}: {len(ev)}")
            cache[str(mmsi)] = {"identity": ident, "events": events,
                                "track_probe": track_probe,
                                "baseline_period": [BASELINE_START, BASELINE_END]}
            time.sleep(0.2)
        Path(DEEP_CACHE_PATH).write_text(json.dumps(cache, indent=1))
        print(f"  cached raw events -> {DEEP_CACHE_PATH}")
    return cache


# ==========================================================================
# Geometry / small helpers
# ==========================================================================
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _ts(s):
    return pd.to_datetime(s, utc=True, errors="coerce")


def _eez_names(codes):
    return [EEZ_LABEL.get(str(c), f"EEZ:{c}") for c in (codes or [])]


# ==========================================================================
# Flatten events -> one long DataFrame per vessel (this is what gets saved)
# ==========================================================================
def events_to_frame(mmsi, blob):
    rows = []
    ev = blob.get("events", {})
    for e in ev.get("port_visit", []):
        pv = e.get("port_visit") or {}
        sa = pv.get("startAnchorage") or {}
        rows.append({
            "event_type": "port_visit",
            "start": e.get("start"), "end": e.get("end"),
            "lat": (e.get("position") or {}).get("lat"),
            "lon": (e.get("position") or {}).get("lon"),
            "duration_hrs": pv.get("durationHrs"),
            "confidence": pv.get("confidence"),
            "port_id": sa.get("id"), "port_name": sa.get("name"),
            "port_flag": sa.get("flag"),
            "eez": ";".join(_eez_names((e.get("regions") or {}).get("eez"))),
            "avg_speed_kn": None,
            "dist_from_shore_km": (e.get("distances") or {}).get("startDistanceFromShoreKm"),
        })
    for e in ev.get("loitering", []):
        lo = e.get("loitering") or {}
        rows.append({
            "event_type": "loitering",
            "start": e.get("start"), "end": e.get("end"),
            "lat": (e.get("position") or {}).get("lat"),
            "lon": (e.get("position") or {}).get("lon"),
            "duration_hrs": lo.get("totalTimeHours"),
            "confidence": None, "port_id": None, "port_name": None, "port_flag": None,
            "eez": ";".join(_eez_names((e.get("regions") or {}).get("eez"))),
            "avg_speed_kn": lo.get("averageSpeedKnots"),
            "dist_from_shore_km": lo.get("averageDistanceFromShoreKm"),
        })
    for e in ev.get("gap", []):
        g = e.get("gap") or {}
        rows.append({
            "event_type": "ais_gap",
            "start": e.get("start"), "end": e.get("end"),
            "lat": (e.get("position") or {}).get("lat"),
            "lon": (e.get("position") or {}).get("lon"),
            "duration_hrs": g.get("durationHours"),
            "confidence": None, "port_id": None, "port_name": None, "port_flag": None,
            "eez": ";".join(_eez_names((e.get("regions") or {}).get("eez"))),
            "avg_speed_kn": g.get("impliedSpeedKnots"),
            "dist_from_shore_km": g.get("distanceFromShoreKm"),
        })
    for e in ev.get("encounter", []):
        en = e.get("encounter") or {}
        rows.append({
            "event_type": "encounter",
            "start": e.get("start"), "end": e.get("end"),
            "lat": (e.get("position") or {}).get("lat"),
            "lon": (e.get("position") or {}).get("lon"),
            "duration_hrs": en.get("medianDurationHours") or en.get("durationHours"),
            "confidence": en.get("authorizationStatus"),
            "port_id": None,
            "port_name": (en.get("vessel") or {}).get("name") if en.get("vessel") else None,
            "port_flag": None,
            "eez": ";".join(_eez_names((e.get("regions") or {}).get("eez"))),
            "avg_speed_kn": en.get("medianSpeedKnots"),
            "dist_from_shore_km": (e.get("distances") or {}).get("startDistanceFromShoreKm"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["start"] = _ts(df["start"])
    df["end"] = _ts(df["end"])
    df = df.sort_values("start").reset_index(drop=True)
    df.insert(0, "mmsi", mmsi)
    df.insert(1, "ship_name", SHORTLIST[mmsi])
    return df


# ==========================================================================
# Baseline (requirement 2)
# ==========================================================================
def build_baseline(mmsi, blob, df):
    ident = blob.get("identity", {})
    out = {"mmsi": mmsi, "ship_name": SHORTLIST[mmsi], "identity": ident,
           "track_probe": blob.get("track_probe")}

    pv = df[df.event_type == "port_visit"].copy()
    lo = df[df.event_type == "loitering"].copy()
    gaps = df[df.event_type == "ais_gap"].copy()
    enc = df[df.event_type == "encounter"].copy()

    # ---- operating area -------------------------------------------------
    pos = df.dropna(subset=["lat", "lon"])
    if not pos.empty:
        out["bbox"] = (round(float(pos.lat.min()), 2), round(float(pos.lat.max()), 2),
                       round(float(pos.lon.min()), 2), round(float(pos.lon.max()), 2))
        out["centroid"] = (round(float(pos.lat.mean()), 2), round(float(pos.lon.mean()), 2))
        # rough linear extent of the operating box, km
        out["extent_km"] = round(haversine_km(pos.lat.min(), pos.lon.min(),
                                              pos.lat.max(), pos.lon.max()), 0)
    else:
        out["bbox"] = out["centroid"] = None
        out["extent_km"] = 0.0

    # ---- port calls: countries, names, cadence ------------------------
    pv_ok = pv.dropna(subset=["start"])
    out["n_port_visits"] = len(pv_ok)
    out["port_flags"] = (pv_ok["port_flag"].value_counts().to_dict())
    out["port_ids"] = (pv_ok["port_id"].value_counts().to_dict())
    out["n_port_countries"] = pv_ok["port_flag"].nunique()
    if len(pv_ok) >= 2:
        gaps_between = pv_ok["start"].sort_values().diff().dt.total_seconds().dropna() / 86400.0
        out["port_interval_days_median"] = round(float(gaps_between.median()), 1)
        out["port_visit_dur_hrs_median"] = round(float(pv_ok["duration_hrs"].median()), 1)
        out["port_visit_dur_hrs_max"] = round(float(pv_ok["duration_hrs"].max()), 1)
    else:
        out["port_interval_days_median"] = None
        out["port_visit_dur_hrs_median"] = None
        out["port_visit_dur_hrs_max"] = None

    # ---- AIS reporting-gap baseline ----------------------------------
    # GFW's own gap detector. Also express raw AIS density from identity.
    out["n_ais_gap_events"] = len(gaps)
    if len(gaps):
        out["ais_gap_hrs_median"] = round(float(gaps["duration_hrs"].median()), 1)
        out["ais_gap_hrs_max"] = round(float(gaps["duration_hrs"].max()), 1)
    else:
        out["ais_gap_hrs_median"] = out["ais_gap_hrs_max"] = 0.0
    pcount = ident.get("positions_counter")
    tfrom, tto = _ts(ident.get("transmission_from")), _ts(ident.get("transmission_to"))
    if pcount and pd.notna(tfrom) and pd.notna(tto) and tto > tfrom:
        days = (tto - tfrom).total_seconds() / 86400.0
        out["ais_positions_per_day_lifetime"] = round(pcount / days, 0)
    else:
        out["ais_positions_per_day_lifetime"] = None

    # ---- speed baseline (two coarse estimates, both labelled) --------
    out["loiter_speed_kn_median"] = (round(float(lo["avg_speed_kn"].median()), 2)
                                     if len(lo) else None)
    out["loiter_speed_kn_p90"] = (round(float(lo["avg_speed_kn"].quantile(0.9)), 2)
                                  if len(lo) else None)
    transit_speeds = []
    p = pv_ok.sort_values("start").reset_index(drop=True)
    for i in range(len(p) - 1):
        a, b = p.loc[i], p.loc[i + 1]
        if pd.isna(a["end"]) or pd.isna(b["start"]):
            continue
        hrs = (b["start"] - a["end"]).total_seconds() / 3600.0
        if hrs < 3:
            continue
        d_km = haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
        if d_km < 30:
            continue
        transit_speeds.append((d_km / hrs) / 1.852)  # knots
    if transit_speeds:
        ts = np.array(transit_speeds)
        out["transit_speed_kn_median"] = round(float(np.median(ts)), 1)
        out["transit_speed_kn_range"] = (round(float(ts.min()), 1), round(float(ts.max()), 1))
        out["n_transit_legs"] = len(ts)
    else:
        out["transit_speed_kn_median"] = None
        out["transit_speed_kn_range"] = None
        out["n_transit_legs"] = 0

    # ---- loitering / encounter baseline -----------------------------
    out["n_loitering"] = len(lo)
    out["loiter_hrs_median"] = round(float(lo["duration_hrs"].median()), 1) if len(lo) else 0.0
    out["loiter_hrs_max"] = round(float(lo["duration_hrs"].max()), 1) if len(lo) else 0.0
    out["n_encounters"] = len(enc)

    # ---- India-area activity (requirement 5) ------------------------
    india_hits = []
    for _, r in df.iterrows():
        why = []
        if isinstance(r["port_flag"], str) and r["port_flag"] == "IND":
            why.append(f"port call flag IND ({r['port_id']})")
        if isinstance(r["port_id"], str) and r["port_id"].startswith("ind-"):
            why.append(f"GFW anchorage id {r['port_id']}")
        if isinstance(r["eez"], str) and "India" in r["eez"]:
            why.append("event inside India EEZ")
        if pd.notna(r["lat"]) and pd.notna(r["lon"]):
            la, lo_, mnla, mxla, mnlo, mxlo = (r["lat"], r["lon"], *INDIA_BBOX)
            if mnla <= la <= mxla and mnlo <= lo_ <= mxlo:
                near = sorted(INDIA_PORTS.items(),
                              key=lambda kv: haversine_km(la, lo_, *kv[1]))[0]
                if haversine_km(la, lo_, *near[1]) < 60:
                    why.append(f"within 60 km of {near[0].replace('_', ' ').title()}")
                elif not why:
                    why.append("position inside Indian-waters box")
        if why:
            india_hits.append({
                "when": r["start"].strftime("%Y-%m-%d") if pd.notna(r["start"]) else "?",
                "event_type": r["event_type"], "lat": r["lat"], "lon": r["lon"],
                "why": "; ".join(sorted(set(why))),
            })
    out["india_activity"] = india_hits
    out["india_found"] = bool(india_hits)

    return out


# ==========================================================================
# Incident-window flagged events (from the saved window outputs)
# ==========================================================================
SPOOF_DISTANCE_KM = 20.0


def load_incident_flags(mmsi):
    """The specific flagged events for this vessel from our 4 windows -- the
    same rows src/vessel_history.py consolidates, but kept per-incident here so
    each can be tested against the baseline."""
    flags = []

    # high-tier trajectory deviation
    try:
        tv = pd.read_csv(f"{PROC}/hormuz_trajectory_deviation_v2.csv")
        tv = tv[(tv["mmsi"] == mmsi) & (tv["tier"] == "high")]
        for _, r in tv.iterrows():
            flags.append({
                "window": r["window"], "check": "trajectory_deviation_high",
                "timestamp": r["predicted_timestamp"],
                "lat": r["actual_lat"], "lon": r["actual_lon"],
                "metric": "deviation_km", "value": round(float(r["deviation_km"]), 1),
                "context": f"gap12={r.get('gap12_hours')}h",
            })
    except FileNotFoundError:
        pass

    # genuine multi-vessel cluster membership
    try:
        sc = pd.read_csv(f"{PROC}/hormuz_spatiotemporal_clusters.csv")
        sc = sc[(sc["vessel_mmsi"] == mmsi)
                & (sc["cluster_coherence_label"] == "genuine_multivessel_coherence")]
        for cid, g in sc.groupby("cluster_id"):
            g = g.sort_values("timestamp")
            r0 = g.iloc[0]
            flags.append({
                "window": r0["window"], "check": "genuine_cluster_member",
                "timestamp": r0["timestamp"], "lat": r0["lat"], "lon": r0["lon"],
                "metric": "cluster_radius_km", "value": float(r0["cluster_radius_km"]),
                "context": f"cluster {cid}, {int(r0['cluster_n_distinct_vessels'])} vessels",
            })
    except FileNotFoundError:
        pass

    # SAR-vs-AIS likely-spoofed
    for path, win in [
        (f"{PROC}/hormuz_crisis_mar2026_reclassified.csv", "hormuz_crisis_mar2026"),
        (f"{PROC}/hormuz_control_mar2026_reclassified.csv", "hormuz_control_jan2026"),
        (f"{PROC}/qatar_gnss_spoofing_oct2025_classified.csv", "qatar_spoofing_oct2025"),
        (f"{PROC}/qatar_control_sept2025_classified.csv", "qatar_control_sep2025"),
    ]:
        try:
            d = pd.read_csv(path)
        except FileNotFoundError:
            continue
        key = d["mmsi"] if "mmsi" in d.columns else pd.Series([np.nan] * len(d))
        if "matched_mmsi" in d.columns:
            key = key.fillna(d["matched_mmsi"])
        d = d[key == mmsi]
        if d.empty:
            continue
        dist = pd.to_numeric(d.get("distance_km"), errors="coerce")
        d = d[(d.get("classification").astype(str) == "likely_spoofed")
              | (dist > SPOOF_DISTANCE_KM)]
        for _, r in d.iterrows():
            flags.append({
                "window": win, "check": "sar_ais_likely_spoofed",
                "timestamp": r.get("sar_timestamp"),
                "lat": r.get("ais_lat"), "lon": r.get("ais_lon"),
                "metric": "sar_ais_distance_km",
                "value": round(float(pd.to_numeric(r.get("distance_km"), errors="coerce")), 1),
                "context": str(r.get("classification")),
            })
    return flags


# ==========================================================================
# Cross-reference + the two-reading verdict (requirements 3 & 4)
# ==========================================================================
def cross_reference(baseline, flags):
    b = baseline
    bbox = b["bbox"]
    checks = []
    inside_area = 0
    for f in flags:
        la, lo_ = f.get("lat"), f.get("lon")
        in_box = None
        if bbox and pd.notna(la) and pd.notna(lo_):
            pad = 0.75  # deg
            in_box = (bbox[0] - pad <= la <= bbox[1] + pad
                      and bbox[2] - pad <= lo_ <= bbox[3] + pad)
            inside_area += int(bool(in_box))
        checks.append({**f, "inside_baseline_area": in_box})
    n_loc = sum(1 for c in checks if c["inside_baseline_area"] is not None)
    frac_inside = (inside_area / n_loc) if n_loc else None

    # ---------- baseline-anomaly scorecard (drives which reading wins) -----
    ident = b["identity"]
    anomaly, clean = [], []

    # identity red flags
    try:
        v3 = pd.read_csv(V3_SCORES)
        row = v3[v3["mmsi"] == b["mmsi"]]
        foc_tier = row["foc_list_matched"].iloc[0] if len(row) else "none"
    except FileNotFoundError:
        foc_tier = "none"
    if foc_tier == "shadow_fleet":
        anomaly.append("flag is on the narrow shadow-fleet reflagging list")
    if ident.get("n_distinct_flags", 1) >= 2:
        flg = " -> ".join(dict.fromkeys(h["flag"] for h in ident.get("identity_history", []) if h["flag"]))
        anomaly.append(f"reflagging history ({ident['n_distinct_flags']} flags: {flg})")
    if ident.get("n_distinct_names", 1) >= 2:
        nms = " -> ".join(dict.fromkeys(h["name"] for h in ident.get("identity_history", []) if h["name"]))
        anomaly.append(f"renamed hull ({nms})")

    # behaviour-pattern anomalies over the 6 months
    if (b.get("port_visit_dur_hrs_max") or 0) > 21 * 24:
        anomaly.append(f"a lay-up of {b['port_visit_dur_hrs_max']/24:.0f} days in one place")
    if b.get("extent_km") is not None and b["extent_km"] < 1500 and (b.get("n_port_countries") or 0) <= 2:
        anomaly.append(f"geographically confined (~{b['extent_km']:.0f} km span, "
                       f"{b.get('n_port_countries')} port country/ies) - no real international voyaging")
    if b.get("n_ais_gap_events", 0) >= 3:
        anomaly.append(f"{b['n_ais_gap_events']} AIS gap events")

    # cleanliness indicators
    if b.get("n_ais_gap_events", 0) == 0:
        clean.append("0 AIS gap events (continuous transmission)")
    if b.get("n_encounters", 0) == 0:
        clean.append("0 ship-to-ship encounters")
    if (b.get("n_port_countries") or 0) >= 4:
        clean.append(f"{b['n_port_countries']} distinct port countries (broad legitimate trade)")
    if b.get("extent_km", 0) and b["extent_km"] > 4000:
        clean.append(f"genuinely international operating area (~{b['extent_km']:.0f} km span)")
    if ident.get("n_distinct_flags", 1) == 1 and ident.get("n_distinct_names", 1) == 1:
        clean.append("single stable identity (no reflag / rename)")
    ppd = b.get("ais_positions_per_day_lifetime")
    if ppd and ppd > 200:
        clean.append(f"dense AIS ({ppd:.0f} positions/day lifetime)")

    # ---------- the two readings ----------
    behaviour_matches_baseline = (frac_inside is not None and frac_inside >= 0.6)
    if not behaviour_matches_baseline:
        verdict = ("OUTSIDE baseline - the flagged behaviour is NOT explained by "
                   "this vessel's own 6-month pattern; treat as a genuine anomaly.")
        better = "outside_baseline"
    else:
        # behaviour DOES match baseline -> do NOT default to "cleared"
        if len(anomaly) >= 2 and len(anomaly) >= len(clean):
            better = "(b) persistent"
            verdict = (
                "INSIDE baseline, but reading (b) is better supported: the "
                "flagged pattern matches a 6-month baseline that is ITSELF "
                "persistently unusual. Not 'cleared' - the suspicious look is "
                "sustained, and combined with the identity red flags this is a "
                "STRONGER finding than a one-off would be.")
        elif len(anomaly) >= 1:
            better = "(a)/(b) mixed"
            verdict = (
                "INSIDE baseline. Reading (a) - ordinary behaviour, sparse-window "
                "false positive - is plausible for the movement itself, BUT "
                f"{len(anomaly)} baseline-anomaly indicator(s) mean reading (b) "
                "cannot be dismissed: the vessel's normal is partly abnormal.")
        else:
            better = "(a) ordinary"
            verdict = (
                "INSIDE baseline and the baseline looks clean on every indicator "
                "checked: reading (a) - genuinely ordinary behaviour for this "
                "vessel, a false positive of the incident-window method - is the "
                "better-supported reading. (Still not proof of nothing; only that "
                "the behavioural signal does not survive a vessel-specific check.)")

    return {
        "flag_checks": checks,
        "frac_inside_baseline_area": frac_inside,
        "behaviour_matches_baseline": behaviour_matches_baseline,
        "anomaly_indicators": anomaly,
        "clean_indicators": clean,
        "better_supported_reading": better,
        "verdict": verdict,
    }


# ==========================================================================
# Report + save
# ==========================================================================
def report_vessel(baseline, flags, xref):
    b = baseline
    ident = b["identity"]
    line = "=" * 96
    print(f"\n{line}\n{b['ship_name']}  (MMSI {b['mmsi']})   VESSEL-SPECIFIC BASELINE COMPARISON")
    print(f"{line}")
    print(f"  baseline period      : {BASELINE_START} -> {BASELINE_END} (continuous)")
    print(f"  GFW vessel_id        : {ident.get('vessel_id')}")
    print(f"  per-vessel track API : {b['track_probe']}  -> baseline built from discrete /events only")
    print(f"  identity             : name={ident.get('name')}  flag={ident.get('flag')}  "
          f"IMO={ident.get('imo')}  AIS {ident.get('transmission_from')}..{ident.get('transmission_to')}")
    if len(ident.get("identity_history", [])) > 1:
        print(f"  identity history     :")
        for h in ident["identity_history"]:
            print(f"      {h['from']}..{h['to']}  ssvid={h['ssvid']}  flag={h['flag']}  "
                  f"name={h['name']}  imo={h['imo']}  ({h['positions']} positions)")

    print(f"\n  -- BASELINE (this vessel's own normal) --")
    print(f"  operating area       : bbox lat[{b['bbox'][0]},{b['bbox'][1]}] lon[{b['bbox'][2]},{b['bbox'][3]}]"
          f"  (~{b['extent_km']:.0f} km span)  centroid {b['centroid']}")
    print(f"  port calls           : {b['n_port_visits']} visits in "
          f"{b['n_port_countries']} country/ies -> {b['port_flags']}")
    top_ports = list(b["port_ids"].items())[:8]
    print(f"                         top ports: {top_ports}")
    print(f"  port cadence         : median {b['port_interval_days_median']} days between calls; "
          f"median stay {b['port_visit_dur_hrs_median']}h; longest stay {b['port_visit_dur_hrs_max']}h")
    print(f"  AIS reporting gaps   : {b['n_ais_gap_events']} GFW gap events "
          f"(median {b['ais_gap_hrs_median']}h, max {b['ais_gap_hrs_max']}h); "
          f"lifetime density ~{b['ais_positions_per_day_lifetime']} pos/day")
    print(f"  speed (no track!)    : at-rest/loiter median {b['loiter_speed_kn_median']}kn "
          f"(p90 {b['loiter_speed_kn_p90']}kn); implied transit "
          f"{b['transit_speed_kn_median']}kn over {b['n_transit_legs']} legs "
          f"range {b['transit_speed_kn_range']}")
    print(f"  loitering / STS      : {b['n_loitering']} loiter events "
          f"(median {b['loiter_hrs_median']}h, max {b['loiter_hrs_max']}h); "
          f"{b['n_encounters']} ship-to-ship encounters")

    print(f"\n  -- INDIA-AREA ACTIVITY (bonus check) --")
    if b["india_found"]:
        print(f"  *** INDIA ACTIVITY FOUND *** ({len(b['india_activity'])} events) -- the eventual India")
        print(f"  phase would start from a vessel already under investigation, not from zero:")
        for h in b["india_activity"][:20]:
            print(f"      {h['when']}  {h['event_type']:11} ({h['lat']:.2f},{h['lon']:.2f})  {h['why']}")
        if len(b["india_activity"]) > 20:
            print(f"      ... +{len(b['india_activity']) - 20} more")
    else:
        print(f"  none - no Indian port call, no Indian EEZ entry, no position inside Indian waters "
              f"in the 6-month baseline.")

    print(f"\n  -- INCIDENT-WINDOW FLAGS vs THIS BASELINE ({len(flags)} flagged events) --")
    for c in xref["flag_checks"]:
        loc = ("inside" if c["inside_baseline_area"] else
               "OUTSIDE" if c["inside_baseline_area"] is False else "n/a")
        print(f"      {str(c['timestamp'])[:16]}  {c['window']:24} {c['check']:26} "
              f"{c['metric']}={c['value']:<8} [{c['context']}]  area:{loc}")
    fi = xref["frac_inside_baseline_area"]
    print(f"  -> {fi:.0%} of locatable flags fall inside this vessel's own operating area"
          if fi is not None else "  -> no locatable flags")

    print(f"\n  -- CIRCULARITY-AWARE READING (both stated; not defaulting to reassurance) --")
    print(f"  baseline-anomaly indicators ({len(xref['anomaly_indicators'])}):")
    for a in xref["anomaly_indicators"] or ["   (none)"]:
        print(f"      + {a}")
    print(f"  baseline-clean indicators ({len(xref['clean_indicators'])}):")
    for a in xref["clean_indicators"] or ["   (none)"]:
        print(f"      - {a}")
    print(f"\n  (a) genuinely ordinary for this vessel  |  (b) persistently suspicious across 6 months")
    print(f"  BETTER SUPPORTED: {xref['better_supported_reading']}")
    for chunk in _wrap(xref["verdict"], 92):
        print(f"    {chunk}")


def _wrap(s, n):
    words, line, out = s.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > n:
            out.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        out.append(line)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-gfw", action="store_true",
                    help="use only the cached raw events (no network)")
    ap.add_argument("--refresh-gfw", action="store_true",
                    help="ignore the deep-events cache and re-pull")
    args = ap.parse_args()

    print("Vessel-specific baseline comparison (NOT cross-window reliability).")
    print(f"Shortlist: {', '.join(f'{v} ({k})' for k, v in SHORTLIST.items())}")
    cache = load_all_events(enabled=not args.no_gfw, refresh=args.refresh_gfw)

    Path(PROC).mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for mmsi in SHORTLIST:
        blob = cache.get(str(mmsi))
        if not blob or not blob.get("events"):
            print(f"\n!! {SHORTLIST[mmsi]} ({mmsi}): no cached events - skipped.")
            continue
        df = events_to_frame(mmsi, blob)
        out_csv = f"{PROC}/vessel_deep_history_{mmsi}.csv"
        df.to_csv(out_csv, index=False)

        baseline = build_baseline(mmsi, blob, df)
        flags = load_incident_flags(mmsi)
        xref = cross_reference(baseline, flags)
        report_vessel(baseline, flags, xref)
        print(f"\n  per-event baseline table ({len(df)} rows) -> {out_csv}")

        summary_rows.append({
            "mmsi": mmsi, "ship_name": SHORTLIST[mmsi],
            "baseline_bbox": baseline["bbox"], "extent_km": baseline["extent_km"],
            "n_port_visits": baseline["n_port_visits"],
            "n_port_countries": baseline["n_port_countries"],
            "n_ais_gap_events": baseline["n_ais_gap_events"],
            "n_encounters": baseline["n_encounters"],
            "india_found": baseline["india_found"],
            "n_flags": len(flags),
            "frac_flags_inside_baseline": xref["frac_inside_baseline_area"],
            "better_supported_reading": xref["better_supported_reading"],
        })

    if summary_rows:
        s = pd.DataFrame(summary_rows)
        print(f"\n{'=' * 96}\nSHORTLIST SUMMARY\n{'=' * 96}")
        print(s.to_string(index=False))


if __name__ == "__main__":
    main()
