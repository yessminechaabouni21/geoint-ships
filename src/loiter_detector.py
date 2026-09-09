"""Loiter / dwell detector -- closes the structural gap found by the
TIBURON / SEASONS I follow-the-suspect stress test.

WHY THIS EXISTS
    A vessel that sits stationary for days near a flagged facility (e.g. a
    sanctioned refinery) is invisible to the existing chain:
      * trajectory_predict.py scores a motionless vessel as ~0 km deviation
        -> tier "normal" (it flags *movement away from* a predicted
        position; a perfectly idle vessel is the opposite of that);
      * spatiotemporal_cluster.py only ingests trajectory high/notable and
        SAR likely_spoofed episodes, so an idle vessel never enters it.
    SEASONS I sat ~9 days at the Vadinar anchorage in jan-feb 2026 (exactly
    the CNBC/Kpler "outside Vadinar, not yet unloaded" report) and produced
    ZERO flags from the whole validated chain. This module detects that.

INPUT (no new API calls)
    The already-pulled GFW AIS presence bins for a window:
    data/raw/{label}_ais_positions.csv -- one row per vessel per ~1.1 km
    grid cell per HOURLY bucket (see src/fetch_ais.py / src/config.py). The
    "timestamp" column is the per-row hourly bucket.

OUTPUT
    data/processed/loiter_flags_{label}.csv -- one row per dwell episode,
    with duration tier, nearest facility, and an `escalate` flag consumed by
    src.watchlist_trigger (new tier-1 category
    "prolonged_dwell_at_flagged_facility").

Run:
    python -m src.loiter_detector jamnagar_vadinar_feb2026
    python -m src.loiter_detector hormuz_control_mar2026 hormuz_crisis_mar2026
"""
import os
import sys

import numpy as np
import pandas as pd

from src.match import haversine_km

RAW = "data/raw"
PROC = "data/processed"

# ==========================================================================
# Thresholds -- documented to the same standard as every other cutoff in
# this project (config.py, match.py, calibrate_thresholds.py).
# ==========================================================================

# A dwell episode is a maximal run of hourly presence bins that all stay
# within DWELL_RADIUS_KM of the run's running centroid (and never gap more
# than GAP_TOL_H). RADIUS, not per-step distance, because a vessel swinging
# at anchor legitimately moves >1 km between consecutive hourly bins (a
# 0.01 deg grid step is ~1.03 km at 22.5N) yet has gone nowhere. 5.0 km is:
#   * larger than the GFW grid cell (~1.1 km) x a few cells of anchor-swing
#     / re-anchoring scatter seen in the Vadinar/Jamnagar anchorage field;
#   * well under the FLAGGED_FACILITY_RADIUS_KM (25 km) so one episode can
#     never bridge two facilities;
#   * below the India-region calibrated eps_spatial (~8.98 km).
DWELL_RADIUS_KM = 5.0

# Maximum present-to-present time gap tolerated *inside* one dwell episode.
# GFW presence coverage has holes even while a vessel is physically there.
# 6.0 h reuses the India-region calibrated eps_temporal_hours (the value
# derived for "sparser AIS coverage" in threshold_calibration_india_
# mangalore.json / _jamnagar_vadinar.json). A gap <= 6 h with the position
# still inside the dwell radius is one continuous dwell, not two.
GAP_TOL_H = 6.0

# Episodes shorter than this are ordinary transient stops (pilot boarding,
# a brief hold, a slow turn near a berth) and are not reported at all.
MIN_DWELL_H = 6.0

# Duration tiers. Reference points, all from THIS project's own data:
#   * TIBURON deep-history baseline (this session): "median stay 44.6 h" ->
#     ~2 days is the top of ordinary port/anchorage queuing.
#   * TIBURON's *typical* Vadinar call = 4.6 days / ~110 h (its 2025-10-23
#     call) -> a normal-for-that-vessel call must land in `extended`, NOT
#     the top tier. (classify_duration(110) == "extended" -- asserted in the
#     validation block.)
#   * SEASONS I own-baseline maximum loiter = 219.5 h / 9.1 days, and its
#     jan-feb 2026 Vadinar anchorage hold = 9+ days -> the exceptional case
#     that must reach `prolonged`.
# The 7-day (168 h) line sits above TIBURON's 4.6-day norm and below
# SEASONS I's 9-day exceptional hold -- the exact separation the stress test
# calls for.
ORDINARY_MAX_H = 48.0      # <= 2 days  : ordinary queuing
EXTENDED_MAX_H = 168.0     # 2 - 7 days : extended dwell
#                          # > 7 days   : PROLONGED

# A dwell episode that covers >= this fraction of the whole window's
# wall-clock span is treated as a RESIDENT / unbounded fixture (a harbour
# tug, a permanently-moored barge, or simply a vessel we never saw arrive or
# leave). Escalation is for a vessel that transited IN, held, and (usually)
# transited out -- a bounded visit -- not for port furniture.
RESIDENT_WINDOW_FRACTION = 0.95

# Proximity radii.
FLAGGED_FACILITY_RADIUS_KM = 25.0    # covers a refinery's outer anchorage field
ORDINARY_ANCHORAGE_RADIUS_KM = 20.0  # same value used in check_anchorage_confound.py

# ==========================================================================
# Facility reference lists.
# ==========================================================================

# Sanctioned / high-scrutiny facilities surfaced in this project's own
# shadow-fleet research (CNBC/Kpler, 2026-02-03; EU 18th sanctions package,
# 2025-07, sanctioned Nayara Energy's Vadinar refinery; UK followed). The
# Vadinar anchorage coordinate is where SEASONS I and TIBURON actually sat
# in the jan-feb 2026 window (GFW anchorage id `ind-vadinar`); the Jamnagar
# (Sikka) marine terminal serves the Reliance refinery just south of it.
# `country`: the ISO-3 flag of vessels for which a dwell here is DOMESTIC
# port infrastructure rather than a foreign visitor holding for a berth.
FLAGGED_FACILITIES = {
    "Vadinar terminal / Nayara refinery (EU+UK sanctioned)": (22.55, 69.65, "IND"),
    "Jamnagar (Sikka) marine terminal": (22.43, 69.72, "IND"),
}

# Ordinary anchorages / ports -- a multi-day wait here is routine berth
# queuing, not a flagged-facility dwell. The Hormuz entries are reused
# verbatim from src/check_anchorage_confound.py so the false-positive
# behaviour is anchored to a list this project already vetted; the two
# Qatar entries cover the qatar_control / qatar_spoofing windows.
ORDINARY_ANCHORAGES = {
    "Khor Fakkan port, UAE": (25.33917, 56.35611),
    "Fujairah/Khor Fakkan STS anchorage": (25.1500, 56.4500),
    "Khasab port, Oman": (26.20333, 56.24944),
    "Larak Island anchorage, Iran": (26.85333, 56.35556),
    "Qeshm Island port, Iran": (26.9581, 56.2719),
    "Bandar Abbas port, Iran": (27.1365, 56.2808),
    "Ras Laffan port, Qatar": (25.9061, 51.5992),
    "Doha port, Qatar": (25.2969, 51.5511),
}

TIER_RANK = {"ordinary": 0, "extended": 1, "prolonged": 2}


# ==========================================================================
def _nearest(lat, lon, table):
    """(name, km) of the closest entry in `table` ((lat, lon[, country])
    tuples), or (None, inf) if empty."""
    best_name, best_km = None, float("inf")
    for name, coords in table.items():
        flat, flon = coords[0], coords[1]
        d = float(haversine_km(lat, lon, flat, flon))
        if d < best_km:
            best_name, best_km = name, d
    return best_name, best_km


def classify_duration(hours):
    if hours <= ORDINARY_MAX_H:
        return "ordinary"
    if hours <= EXTENDED_MAX_H:
        return "extended"
    return "prolonged"


def _dwell_episodes_for_vessel(g):
    """g: one vessel's presence rows, sorted by timestamp. Yield a dict per
    dwell episode: a maximal run of consecutive bins whose members all stay
    within DWELL_RADIUS_KM of the run's running centroid and never gap more
    than GAP_TOL_H."""
    g = g.sort_values("timestamp")
    ts = g["timestamp"].to_numpy()
    lat = g["lat"].to_numpy(dtype=float)
    lon = g["lon"].to_numpy(dtype=float)
    n = len(g)

    episodes = []
    i = 0
    while i < n:
        members = [i]
        clat, clon = lat[i], lon[i]
        j = i + 1
        while j < n:
            gap_h = (pd.Timestamp(ts[j]) - pd.Timestamp(ts[j - 1])).total_seconds() / 3600.0
            if gap_h > GAP_TOL_H:
                break
            cand_lat = lat[list(members) + [j]]
            cand_lon = lon[list(members) + [j]]
            nclat, nclon = float(cand_lat.mean()), float(cand_lon.mean())
            if float(np.max(haversine_km(nclat, nclon, cand_lat, cand_lon))) > DWELL_RADIUS_KM:
                break
            members.append(j)
            clat, clon = nclat, nclon
            j += 1

        a, b = members[0], members[-1]
        t0, t1 = pd.Timestamp(ts[a]), pd.Timestamp(ts[b])
        dur_h = (t1 - t0).total_seconds() / 3600.0
        if dur_h >= MIN_DWELL_H:
            seg_lat, seg_lon = lat[a:b + 1], lon[a:b + 1]
            radius_km = float(np.max(haversine_km(clat, clon, seg_lat, seg_lon)))
            nb = b - a + 1
            episodes.append({
                "episode_start": t0, "episode_end": t1,
                "duration_h": round(dur_h, 2), "n_bins": int(nb),
                # continuity = present bins / wall-clock hours (1.0 == hourly)
                "continuity": round(nb / (dur_h + 1.0), 3),
                "centroid_lat": round(clat, 5), "centroid_lon": round(clon, 5),
                "radius_km": round(radius_km, 3),
            })
        i = b + 1 if b > i else i + 1
    return episodes


def detect(label):
    """Return the loiter-flags DataFrame for one window label."""
    ais = pd.read_csv(f"{RAW}/{label}_ais_positions.csv", parse_dates=["timestamp"])
    ais = ais.dropna(subset=["mmsi", "lat", "lon", "timestamp"])
    ais["mmsi"] = ais["mmsi"].astype("int64")

    win_start, win_end = ais["timestamp"].min(), ais["timestamp"].max()
    win_span_h = max((win_end - win_start).total_seconds() / 3600.0, 1.0)

    rows = []
    for mmsi, g in ais.groupby("mmsi"):
        name = str(g["ship_name"].dropna().iloc[0]) if g["ship_name"].notna().any() else None
        flag = str(g["flag"].dropna().iloc[0]) if g["flag"].notna().any() else None

        for ep in _dwell_episodes_for_vessel(g):
            tier = classify_duration(ep["duration_h"])
            f_name, f_km = _nearest(ep["centroid_lat"], ep["centroid_lon"], FLAGGED_FACILITIES)
            a_name, a_km = _nearest(ep["centroid_lat"], ep["centroid_lon"], ORDINARY_ANCHORAGES)
            near_flagged = f_km <= FLAGGED_FACILITY_RADIUS_KM
            near_ordinary = a_km <= ORDINARY_ANCHORAGE_RADIUS_KM

            if near_flagged:
                loc_class, loc_bonus = "flagged_facility", 2
                nearest_name, nearest_km = f_name, f_km
                facility_country = FLAGGED_FACILITIES[f_name][2]
            elif near_ordinary:
                loc_class, loc_bonus = "ordinary_anchorage", 0
                nearest_name, nearest_km = a_name, a_km
                facility_country = None
            else:
                loc_class, loc_bonus = "open_water", 1
                nearest_name, nearest_km = (f_name, f_km) if f_km < a_km else (a_name, a_km)
                facility_country = None

            # bounded visit vs resident fixture
            resident = ep["duration_h"] >= RESIDENT_WINDOW_FRACTION * win_span_h
            # domestic port infrastructure: vessel flies the facility's own
            # flag (an India-flagged craft sitting at an Indian terminal is
            # a tug / barge / service boat, not a foreign vessel holding for
            # a discharge berth).
            domestic = (facility_country is not None and flag is not None
                        and flag.upper() == facility_country)

            score = TIER_RANK[tier] + loc_bonus
            if not resident:
                score += 1                      # a bounded visit is more notable than a fixture
            if loc_class == "flagged_facility" and not domestic:
                score += 1                      # foreign vessel holding at a sanctioned facility

            if tier == "prolonged" and loc_class == "flagged_facility" \
                    and not resident and not domestic:
                trigger, escalate = "prolonged_dwell_at_flagged_facility", True
            elif tier == "prolonged" and loc_class == "open_water" and not resident:
                # prolonged idle in open water is STS / holding territory --
                # recorded and surfaced, not auto-escalated without a
                # facility nexus.
                trigger, escalate = "prolonged_dwell_open_water", False
            else:
                trigger, escalate = "", False

            rows.append({
                "mmsi": mmsi, "ship_name": name, "flag": flag, "window": label,
                **ep,
                "dwell_tier": tier,
                "location_class": loc_class,
                "nearest_facility": nearest_name,
                "nearest_facility_km": round(nearest_km, 2),
                "resident_fixture": resident,
                "domestic_flag": domestic,
                "dwell_score": score,
                "trigger_category": trigger,
                "escalate": escalate,
            })

    cols = ["mmsi", "ship_name", "flag", "window", "episode_start", "episode_end",
            "duration_h", "n_bins", "continuity", "centroid_lat", "centroid_lon",
            "radius_km", "dwell_tier", "location_class", "nearest_facility",
            "nearest_facility_km", "resident_fixture", "domestic_flag",
            "dwell_score", "trigger_category", "escalate"]
    df = pd.DataFrame(rows, columns=cols)
    if len(df):
        df = df.sort_values(["escalate", "dwell_score", "duration_h"],
                            ascending=[False, False, False]).reset_index(drop=True)
    return df


def write(label):
    df = detect(label)
    out = f"{PROC}/loiter_flags_{label}.csv"
    df.to_csv(out, index=False)
    return df, out


# ==========================================================================
# Consumed by src.watchlist_trigger
# ==========================================================================
def load_loiter_events(window_key):
    """Long DataFrame (mmsi, window, source, observed, flagged_events,
    event_type) for the ESCALATING dwell episodes of `window_key`, matching
    the shape src.vessel_history.load_events() emits so watchlist_trigger can
    concatenate it. Empty (correct columns) if the flags file is absent."""
    path = f"{PROC}/loiter_flags_{window_key}.csv"
    cols = ["mmsi", "window", "source", "observed", "flagged_events", "event_type"]
    if not os.path.exists(path):
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path)
    esc = df[df["escalate"] == True]  # noqa: E712
    out = []
    for mmsi, g in esc.groupby("mmsi"):
        out.append({
            "mmsi": int(mmsi), "window": window_key,
            "source": f"loiter_flags_{window_key}.csv",
            "observed": True,
            "flagged_events": int(len(g)),   # one per prolonged-dwell episode
            "event_type": "prolonged_dwell_at_flagged_facility",
        })
    return pd.DataFrame(out, columns=cols)


def name_map(window_key):
    """mmsi -> ship_name from a window's loiter-flags file."""
    path = f"{PROC}/loiter_flags_{window_key}.csv"
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path).dropna(subset=["ship_name"])
    return {int(m): str(n) for m, n in zip(df["mmsi"], df["ship_name"])}


# ==========================================================================
def _print_summary(label, df):
    print(f"\n{'=' * 92}\nLOITER / DWELL DETECTOR -- {label}\n{'=' * 92}")
    if df.empty:
        print("  no dwell episodes >= %.0f h" % MIN_DWELL_H)
        return
    by_tier = df["dwell_tier"].value_counts().to_dict()
    print(f"  dwell episodes (>= {MIN_DWELL_H:.0f} h): {len(df)}   by tier: "
          + ", ".join(f"{k}={by_tier.get(k, 0)}"
                      for k in ("ordinary", "extended", "prolonged")))
    print("  by location: " + ", ".join(
        f"{k}={int((df['location_class'] == k).sum())}"
        for k in ("flagged_facility", "ordinary_anchorage", "open_water")))
    prol = df[df["dwell_tier"] == "prolonged"]
    print(f"  prolonged (> 7 d): {len(prol)}  "
          f"[resident fixtures: {int(prol['resident_fixture'].sum())}, "
          f"domestic-flag at flagged facility: {int(prol['domestic_flag'].sum())}]")

    show = ["mmsi", "ship_name", "flag", "episode_start", "episode_end",
            "duration_h", "continuity", "dwell_tier", "nearest_facility",
            "nearest_facility_km", "dwell_score"]
    esc = df[df["escalate"] == True]  # noqa: E712
    print(f"\n  >>> ESCALATING (prolonged_dwell_at_flagged_facility): {len(esc)}")
    if len(esc):
        print(esc[show].to_string(index=False))

    ow = df[(df["dwell_tier"] == "prolonged") & (df["location_class"] == "open_water")
            & (~df["resident_fixture"])]
    if len(ow):
        print(f"\n  prolonged dwell in OPEN WATER (recorded, not auto-escalated): {len(ow)}")
        print(ow[show].to_string(index=False))

    supp = prol[(prol["location_class"] == "flagged_facility")
                & (prol["resident_fixture"] | prol["domestic_flag"])]
    if len(supp):
        print(f"\n  prolonged at a flagged facility but SUPPRESSED "
              f"(resident fixture / domestic-flag harbour craft): {len(supp)}")
        print(supp[show + ["resident_fixture", "domestic_flag"]].to_string(index=False))


def _self_check():
    assert classify_duration(20) == "ordinary"
    assert classify_duration(44.6) == "ordinary"      # TIBURON baseline median stay
    assert classify_duration(110) == "extended"       # TIBURON typical 4.6-day Vadinar call
    assert classify_duration(168) == "extended"
    assert classify_duration(219.5) == "prolonged"    # SEASONS I own-baseline max loiter
    assert classify_duration(221) == "prolonged"      # SEASONS I jan-feb 2026 hold


def main(argv):
    _self_check()
    if not argv:
        print("usage: python -m src.loiter_detector <window_label> [<window_label> ...]")
        return
    for label in argv:
        df, out = write(label)
        _print_summary(label, df)
        print(f"\n  written: {out}  ({len(df)} row(s))")


if __name__ == "__main__":
    main(sys.argv[1:])
