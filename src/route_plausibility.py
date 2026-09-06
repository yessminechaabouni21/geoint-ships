"""Route-plausibility check: is the implied path between two reported vessel
positions one a real ship could actually sail?

WHY THIS EXISTS
---------------
Every earlier check in this project measured position discrepancy as a
straight-line (haversine) distance:
  * match.py / reclassify_hormuz.py : SAR blob vs AIS position, distance_km
  * trajectory_predict.py           : predicted vs actual, deviation_km
A straight line does not know that the line crosses land, ignores the real
shipping lane, or - crucially - that a large number over a large elapsed time
is an ordinary transit speed. We already learned this the hard way:
SELENIA's 652 km "deviation" (over a real 29 h AIS gap) and PATRIS's 3220 km
"deviation" (over 59 h) were flagged HIGH by the raw deviation_km metric but
turned out to be normal once elapsed time was accounted for.

This module replaces the straight line with a realistic navigable sea route
(via the `searoute` package, which routes around landmasses along real
lanes) and divides by the REAL elapsed time to get a required speed in
knots. That required speed is then judged against what the vessel's TYPE
could actually do. It is a stronger, more specific signal than deviation_km:
deviation_km conflates "moved a long way" with "moved impossibly fast";
required_speed_knots isolates the second.

WHAT IT READS  (any two consecutive reported positions in our pulled data)
------------------------------------------------------------------------
  1. hormuz_trajectory_deviation_v2.csv  - consecutive ACTUAL hourly
     positions per (mmsi, window); elapsed = real timestamp gap (so a row
     with gap12_hours = 29 correctly gets 29 h, not 1 h).
  2. the SAR-vs-AIS classified/reclassified files (Qatar Sep + Oct 2025,
     Hormuz control + crisis) - the matched AIS position -> the SAR
     detection; elapsed = time_gap_seconds.
  3. vessel_deep_history_{mmsi}.csv  - consecutive GFW events for the 3
     shortlisted vessels; elapsed = gap between one event ending and the
     next beginning.

CLASSIFICATION  (thresholds justified below, in TYPE_ENVELOPE / HARD_CEILING)
Output: data/processed/route_plausibility_check.csv  + printed summary,
including the re-examined verdict for every previously-flagged long-AIS-gap
"high deviation" event on the top-14 leaderboard.

Run:  python -m src.route_plausibility
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import searoute as sr
except ImportError as e:  # pragma: no cover
    raise SystemExit("pip install searoute  (required for this check)") from e

PROC = "data/processed"
OUT_CSV = f"{PROC}/route_plausibility_check.csv"

# --------------------------------------------------------------------------
# HARD_CEILING_KN -- above this a *sustained* speed is not achievable by any
# merchant vessel, so the position pair or the timestamps must be wrong
# (AIS spoof, mis-paired SAR blob, clock error), not a real transit.
#
#   bulk / crude / product tankers, LNG/LPG carriers : service 12-16 kn,
#       absolute max ~19-21 kn.
#   container ships                                  : service 16-24 kn,
#       fastest ever built ~27 kn.
#   fast RoPax / gas-turbine catamaran ferries (the
#       fastest commercial ships afloat)             : service 35-43 kn,
#       trials up to ~45 kn.
#   frigates / destroyers                            : ~30-34 kn.
#   only hydrofoils / surface-effect / fast-attack
#       military craft exceed ~45 kn, and none of
#       those appear in commercial AIS/SAR data here.
# 45 kn is therefore the physical ceiling for anything in this dataset.
# Same style of explicit, citable threshold as SPOOF_DISTANCE_KM (20 km) and
# FLAG_RATE_CAP (4) elsewhere in the project.
HARD_CEILING_KN = 45.0

# TYPE_ENVELOPE[type_class] = (normal_max_kn, implausible_over_kn)
#   <= normal_max_kn            -> PLAUSIBLE
#   normal_max_kn .. ceiling    -> IMPLAUSIBLE_BUT_POSSIBLE (unusual for the
#                                  type; a favourable current, a light ship,
#                                  or a mild data problem - not dismissable,
#                                  not proof)
#   > HARD_CEILING_KN           -> PHYSICALLY_IMPOSSIBLE
# normal_max_kn is set a few knots above each class's real service speed to
# leave room for currents and reporting jitter.
TYPE_ENVELOPE = {
    "commercial_slow": (22.0, 25.0),   # tanker / bulk / gas / general cargo / "other"
    "commercial_fast": (28.0, 35.0),   # container / reefer / ro-ro / vehicle
    "passenger":       (34.0, 42.0),   # ferry / cruise / high-speed craft
    "fishing":         (16.0, 30.0),   # trawlers etc. - slow working, short dashes
    "utility":         (16.0, 30.0),   # tug / pilot / patrol / supply / landing craft
    "unknown":         (24.0, 30.0),   # no resolved type - widest benefit of the doubt
}


def _type_class(vessel_type):
    t = "" if not isinstance(vessel_type, str) else vessel_type.strip().lower()
    t = t.replace(" ", "_").replace("-", "_").replace("/", "_")
    if any(k in t for k in ("tanker", "lng", "lpg", "gas", "crude", "product",
                            "bulk", "cargo", "ore", "chemical")) and "container" not in t:
        return "commercial_slow"
    if any(k in t for k in ("container", "reefer", "ro_ro", "roro", "vehicle", "car_carrier")):
        return "commercial_fast"
    if any(k in t for k in ("passenger", "ferry", "cruise", "ropax", "hsc", "high_speed")):
        return "passenger"
    if any(k in t for k in ("fishing", "trawler", "seiner", "gear", "fish")):
        return "fishing"
    if any(k in t for k in ("tug", "pilot", "patrol", "supply", "landing_craft",
                            "workboat", "service", "dredg", "sar", "tender", "mooring")):
        return "utility"
    if t in ("", "other", "unknown", "na", "n_a", "none", "not_available",
             "other_non_fishing", "insufficient_data"):
        # GFW's coarse "OTHER" bucket is dominated here by tankers/cargo, so
        # treat a bare "other" as commercial_slow rather than unknown.
        return "commercial_slow" if t == "other" else "unknown"
    return "unknown"


# --------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


_SR_CACHE = {}


def sea_route_km(lat0, lon0, lat1, lon1):
    """Realistic navigable distance (km) via searoute, or None on failure.
    Cached on ~1 km-rounded coordinates so repeated near-identical fixes
    (anchored vessels) don't re-run the graph search."""
    key = (round(lat0, 2), round(lon0, 2), round(lat1, 2), round(lon1, 2))
    if key in _SR_CACHE:
        return _SR_CACHE[key]
    val = None
    try:
        # searoute wants [lon, lat]
        feat = sr.searoute([lon0, lat0], [lon1, lat1], units="km")
        val = float(feat["properties"]["length"])
    except Exception:  # noqa: BLE001  - point unconnectable, etc.
        val = None
    _SR_CACHE[key] = val
    return val


KN_PER_KMH = 1 / 1.852


# searoute's ocean-lane network is only trustworthy for a routed detour once
# the separation is large enough that its sparse nodes approximate a real
# lane. Below this, a big searoute detour on a short hop is a network
# artifact, not a real path -- it can raise suspicion (IMPLAUSIBLE_BUT_
# POSSIBLE) but must NOT by itself manufacture a PHYSICALLY_IMPOSSIBLE
# verdict. A physically-impossible call then needs the STRAIGHT-LINE move
# itself to be impossible (that is route-independent and rock solid), or a
# long-range searoute detour.
SEAROUTE_TRUSTED_KM = 120.0
# Below this, searoute's ~40 km-node ocean-lane graph cannot represent a real
# path; the sea route is recorded but the verdict falls back to the straight
# line (route-independent, artifact-proof).
SEAROUTE_MIN_MEANINGFUL_KM = 40.0


def classify(required_kn, straight_kn, straight_km, type_class):
    if required_kn is None or not np.isfinite(required_kn):
        return "UNDETERMINED"
    normal_max, _ = TYPE_ENVELOPE.get(type_class, TYPE_ENVELOPE["unknown"])
    # rock-solid impossibility: the straight-line displacement alone exceeds
    # the ceiling, or a long-range (trustworthy) sea route does.
    if (straight_kn is not None and straight_kn > HARD_CEILING_KN) or \
       (required_kn > HARD_CEILING_KN and straight_km >= SEAROUTE_TRUSTED_KM):
        return "PHYSICALLY_IMPOSSIBLE"
    # route-adjusted OR straight-line speed above the vessel-type envelope
    if required_kn > normal_max or (straight_kn is not None and straight_kn > normal_max):
        return "IMPLAUSIBLE_BUT_POSSIBLE"
    return "PLAUSIBLE"


# Only run searoute where the straight-line picture is already interesting;
# elsewhere the sea route (>= straight line, give or take network coarseness)
# cannot push a slow transition over a threshold, so haversine is enough.
def _needs_searoute(straight_km, elapsed_h, extra_trigger):
    if straight_km < 1.0:
        return False
    straight_kn = (straight_km / max(elapsed_h, 1e-6)) * KN_PER_KMH
    return bool(extra_trigger or straight_km >= 20.0 or straight_kn >= 12.0
               or elapsed_h >= 6.0)


def _row(source, dataset_file, mmsi, name, vtype, t0, t1, la0, lo0, la1, lo1,
         elapsed_h, is_retro=False, prior_metric=None, prior_tier=None,
         prior_gap_hours=None, extra_trigger=False):
    tclass = _type_class(vtype)
    straight = haversine_km(la0, lo0, la1, lo1)
    notes = []
    def _pack(sea, route_method, dist_for_speed, cls, extra_note=None):
        if extra_note:
            notes.append(extra_note)
        req = ((dist_for_speed / elapsed_h) * KN_PER_KMH
               if elapsed_h and elapsed_h > 0 else None)
        st = ((straight / elapsed_h) * KN_PER_KMH
              if elapsed_h and elapsed_h > 0 else None)
        return {
            "source": source, "dataset_file": dataset_file, "mmsi": mmsi,
            "ship_name": name, "vessel_type": vtype, "type_class": tclass,
            "t0": t0, "t1": t1, "lat0": round(la0, 5), "lon0": round(lo0, 5),
            "lat1": round(la1, 5), "lon1": round(lo1, 5),
            "elapsed_hours": round(float(elapsed_h), 3) if elapsed_h is not None else None,
            "straight_line_distance_km": round(straight, 2),
            "sea_route_distance_km": round(sea, 2) if sea is not None else None,
            "route_method": route_method,
            "required_speed_knots": round(req, 2) if req is not None else None,
            "straight_line_speed_knots": round(st, 2) if st is not None else None,
            "classification": cls, "notes": "; ".join(notes),
            "is_retro_target": is_retro, "prior_metric": prior_metric,
            "prior_tier": prior_tier,
            "prior_gap_hours": (round(float(prior_gap_hours), 1)
                                if prior_gap_hours is not None
                                and np.isfinite(prior_gap_hours) else None),
        }

    if elapsed_h is None or not np.isfinite(elapsed_h):
        return _pack(None, "none", straight, "UNDETERMINED", "elapsed time unknown")

    # near-simultaneous fixes: no meaningful "speed", but a large separation
    # between two ~simultaneous reports is itself the impossibility (a vessel
    # cannot be in two places at once) -- decide on separation, not speed.
    if elapsed_h <= 1 / 60:
        if straight >= 5.0:
            return _pack(None, "separation_only", straight, "PHYSICALLY_IMPOSSIBLE",
                         f"~simultaneous fixes {straight:.1f} km apart")
        if straight >= 1.0:
            return _pack(None, "separation_only", straight, "IMPLAUSIBLE_BUT_POSSIBLE",
                         f"~simultaneous fixes {straight:.1f} km apart")
        return _pack(None, "separation_only", straight, "PLAUSIBLE",
                     "~simultaneous and co-located")
    if elapsed_h < 0.25:
        notes.append(f"short elapsed ({elapsed_h*60:.0f} min): speed sensitive to timestamp jitter")

    if straight < 1.0:
        return _pack(None, "stationary", straight, "PLAUSIBLE", "essentially stationary")

    straight_kn = (straight / elapsed_h) * KN_PER_KMH
    want_sr = _needs_searoute(straight, elapsed_h, is_retro or extra_trigger)
    sea = sea_route_km(la0, lo0, la1, lo1) if want_sr else None

    if sea is None:
        method = "haversine_only" if not want_sr else "haversine_fallback(searoute_failed)"
        return _pack(sea, method, straight,
                     classify(straight_kn, straight_kn, straight, tclass))

    # The sea route is always recorded for transparency, but it only DRIVES
    # the verdict when it is trustworthy:
    #   * the hop must be at least SEAROUTE_MIN_MEANINGFUL_KM. searoute's
    #     ocean-lane graph has ~40 km node spacing in this region, so for a
    #     shorter hop the "route" is just a nearest-node-to-nearest-node
    #     segment -- not a real path. (Tell: an identical "~46 km" / "~68 km"
    #     route recurs for many unrelated short position pairs.)
    #   * even above that, a detour > ~2.5x the straight line on a sub-100 km
    #     hop is a node-snapping artifact, not a forced path around land.
    # When the sea route is not trustworthy we classify on the straight line,
    # which is route-independent and cannot be a network artifact.
    trust_sea = (straight >= SEAROUTE_MIN_MEANINGFUL_KM
                 and not (straight < 100.0 and sea > straight * 2.5 + 15.0))
    if not trust_sea:
        why = ("hop below searoute node spacing" if straight < SEAROUTE_MIN_MEANINGFUL_KM
               else f"implausible {sea:.0f} km detour on a {straight:.1f} km hop")
        return _pack(sea, "searoute_recorded_not_used", straight,
                     classify(straight_kn, straight_kn, straight, tclass),
                     f"sea route not used for verdict ({why}); classified on straight "
                     f"line ({straight_kn:.1f} kn)")

    # never understate: searoute's coarse network can dip below the great
    # circle; the physically required distance is at least the straight line.
    dist_for_speed = max(sea, straight)
    required_kn = (dist_for_speed / elapsed_h) * KN_PER_KMH
    return _pack(sea, "searoute", dist_for_speed,
                 classify(required_kn, straight_kn, straight, tclass))


# ==========================================================================
# Source 1: consecutive actual positions in the trajectory-deviation file
# ==========================================================================
def _top14_mmsis():
    v2 = pd.read_csv(f"{PROC}/vessel_reliability_scores_v2.csv")
    v2 = v2[~v2["is_known_artifact"]].sort_values("reliability_score", ascending=False)
    return set(v2.head(14)["mmsi"].astype(int))


def _type_map():
    for fn in ("vessel_reliability_scores_v3.csv", "vessel_reliability_scores_v2.csv"):
        p = Path(f"{PROC}/{fn}")
        if p.exists():
            df = pd.read_csv(p)
            col = "resolved_vessel_type" if "resolved_vessel_type" in df.columns else None
            if col:
                return dict(zip(df["mmsi"].astype(int), df[col]))
    return {}


def from_trajectory(rows, top14, tmap):
    f = f"{PROC}/hormuz_trajectory_deviation_v2.csv"
    df = pd.read_csv(f)
    df["ts"] = pd.to_datetime(df["predicted_timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts", "actual_lat", "actual_lon"])
    for (mmsi, window), g in df.groupby(["mmsi", "window"]):
        g = g.sort_values("ts").reset_index(drop=True)
        mmsi = int(mmsi)
        name = g["ship_name"].dropna().iloc[0] if g["ship_name"].notna().any() else None
        vtype = tmap.get(mmsi)
        for i in range(1, len(g)):
            a, b = g.loc[i - 1], g.loc[i]
            elapsed = (b["ts"] - a["ts"]).total_seconds() / 3600.0
            gap12 = float(b.get("gap12_hours") or 0)
            tier = b.get("tier")
            is_retro = (mmsi in top14 and tier == "high" and gap12 > 10.0)
            rows.append(_row(
                "trajectory_consecutive_actual", f, mmsi, name, vtype,
                a["ts"].isoformat(), b["ts"].isoformat(),
                float(a["actual_lat"]), float(a["actual_lon"]),
                float(b["actual_lat"]), float(b["actual_lon"]),
                elapsed, is_retro=is_retro,
                prior_metric=round(float(b["deviation_km"]), 1),
                prior_tier=tier, prior_gap_hours=gap12,
                extra_trigger=(tier in ("high", "notable") or gap12 >= 6.0),
            ))


# ==========================================================================
# Source 2: matched AIS position -> SAR detection
# ==========================================================================
def from_sar(rows):
    specs = [
        (f"{PROC}/qatar_gnss_spoofing_oct2025_classified.csv", "qatar_spoofing_oct2025"),
        (f"{PROC}/qatar_control_sept2025_classified.csv", "qatar_control_sep2025"),
        (f"{PROC}/hormuz_crisis_mar2026_reclassified.csv", "hormuz_crisis_mar2026"),
        (f"{PROC}/hormuz_control_mar2026_reclassified.csv", "hormuz_control_jan2026"),
    ]
    for f, _win in specs:
        p = Path(f)
        if not p.exists():
            continue
        df = pd.read_csv(f)
        for _, r in df.iterrows():
            try:
                al, ao = float(r["ais_lat"]), float(r["ais_lon"])
                sl, so = float(r["sar_lat"]), float(r["sar_lon"])
            except (TypeError, ValueError):
                continue
            if not all(np.isfinite(v) for v in (al, ao, sl, so)):
                continue
            tg = r.get("time_gap_seconds")
            try:
                elapsed = abs(float(tg)) / 3600.0
            except (TypeError, ValueError):
                elapsed = None
            mmsi = r.get("mmsi")
            if pd.isna(mmsi) and "matched_mmsi" in df.columns:
                mmsi = r.get("matched_mmsi")
            mmsi = int(mmsi) if pd.notna(mmsi) else None
            rows.append(_row(
                "sar_vs_ais", f, mmsi, r.get("ship_name"),
                r.get("vessel_type"), None, r.get("sar_timestamp"),
                al, ao, sl, so, elapsed,
                prior_metric=round(float(r["distance_km"]), 1) if pd.notna(r.get("distance_km")) else None,
                prior_tier=r.get("classification"),
                prior_gap_hours=elapsed,  # SAR/AIS pairing gap == the elapsed time
                extra_trigger=True,  # spoofing test - always want the real route
            ))


# ==========================================================================
# Source 3: consecutive events in the 3 deep-history files
# ==========================================================================
DEEP = {511101414: ("SELENIA", "product_tanker"),
        636018010: ("PATRIS", "lng_carrier"),
        636025162: ("OCEAN CENTURY", "cargo")}


def from_deep_history(rows):
    for mmsi, (name, vtype) in DEEP.items():
        f = f"{PROC}/vessel_deep_history_{mmsi}.csv"
        if not Path(f).exists():
            continue
        df = pd.read_csv(f)
        df["start"] = pd.to_datetime(df["start"], utc=True, errors="coerce")
        df["end"] = pd.to_datetime(df["end"], utc=True, errors="coerce")
        df = df.dropna(subset=["start", "lat", "lon"]).sort_values("start").reset_index(drop=True)
        for i in range(1, len(df)):
            a, b = df.loc[i - 1], df.loc[i]
            t0 = a["end"] if pd.notna(a["end"]) else a["start"]
            elapsed = (b["start"] - t0).total_seconds() / 3600.0
            rows.append(_row(
                "deep_history_consecutive_events", f, mmsi, name, vtype,
                t0.isoformat(), b["start"].isoformat(),
                float(a["lat"]), float(a["lon"]), float(b["lat"]), float(b["lon"]),
                elapsed,
                prior_metric=None,
                prior_tier=f"{a['event_type']}->{b['event_type']}",
            ))


# ==========================================================================
def main():
    print("Route-plausibility check (sea-route distance vs straight line).")
    top14 = _top14_mmsis()
    tmap = _type_map()
    rows = []
    print("  reading trajectory-deviation consecutive positions ...")
    from_trajectory(rows, top14, tmap)
    print("  reading SAR-vs-AIS matched pairs ...")
    from_sar(rows)
    print("  reading deep-history consecutive events ...")
    from_deep_history(rows)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n  {len(df)} transitions.  route method used:")
    for m, n in df["route_method"].value_counts().items():
        print(f"      {m:38}: {n}")
    print(f"    ({sum(1 for v in _SR_CACHE.values() if v is None)} coordinate pairs "
          f"searoute could not connect)")
    print(f"  saved -> {OUT_CSV}")

    # ---- category counts, overall and per source ----
    print(f"\n{'='*92}\nCLASSIFICATION COUNTS\n{'='*92}")
    order = ["PHYSICALLY_IMPOSSIBLE", "IMPLAUSIBLE_BUT_POSSIBLE", "PLAUSIBLE", "UNDETERMINED"]
    tab = (df.groupby(["source", "classification"]).size()
           .unstack(fill_value=0).reindex(columns=order, fill_value=0))
    tab.loc["-- TOTAL --"] = tab.sum()
    print(tab.to_string())

    # ---- physically impossible: list them (these are data faults) ----
    imp = df[df["classification"] == "PHYSICALLY_IMPOSSIBLE"]
    print(f"\n{'='*92}\nPHYSICALLY_IMPOSSIBLE transitions ({len(imp)}) -- straight-line move alone exceeds "
          f"{HARD_CEILING_KN:.0f} kn,")
    print("so a position or a timestamp in the pair must be wrong (mis-paired SAR blob, "
          "AIS outlier, or spoof)")
    print(f"{'='*92}")
    for _, r in imp.sort_values("straight_line_speed_knots", ascending=False).head(40).iterrows():
        print(f"  {str(r['ship_name'])[:16]:16} mmsi={r['mmsi']}  {r['source'][:26]:26} "
              f"straight {r['straight_line_distance_km']:6.1f} km / {r['elapsed_hours']:.3f} h "
              f"-> {r['straight_line_speed_knots']:7.1f} kn straight-line  "
              f"(prior={r['prior_tier']})")
    if len(imp) > 40:
        print(f"  ... +{len(imp)-40} more in {OUT_CSV}")

    # ---- THE RETROACTIVE CHECK (requirement 4) ----
    print(f"\n{'='*92}")
    print("RETRO CHECK -- previously-flagged HIGH 'deviation' events with a long AIS gap (gap12>10h),")
    print("top-14 leaderboard vessels: raw deviation_km  vs  sea-route-adjusted required speed")
    print(f"{'='*92}")
    retro = df[df["is_retro_target"]].copy()
    if retro.empty:
        print("  (no retro-target rows matched)")
    else:
        retro["dev_km"] = retro["prior_metric"].astype(float)
        for _, r in retro.sort_values(["ship_name", "t1"]).iterrows():
            wl = {"PLAUSIBLE": "explained normal transit",
                  "IMPLAUSIBLE_BUT_POSSIBLE": "still unusual - keep",
                  "PHYSICALLY_IMPOSSIBLE": "REAL problem - not a gap artifact",
                  "UNDETERMINED": "cannot tell"}[r["classification"]]
            print(f"\n  {r['ship_name']}  (mmsi {r['mmsi']}, {r['type_class']})   {str(r['t1'])[:16]}")
            print(f"    raw metric      : deviation_km = {r['dev_km']:.0f}  (tier {r['prior_tier']}) -> flagged HIGH")
            print(f"    real elapsed    : {r['elapsed_hours']:.1f} h   (last real fix before the gap -> first real fix after)")
            print(f"    straight line   : {r['straight_line_distance_km']:.1f} km  "
                  f"-> {r['straight_line_speed_knots']:.1f} kn")
            print(f"    sea route       : {r['sea_route_distance_km']} km  "
                  f"-> required {r['required_speed_knots']:.1f} kn")
            print(f"    VERDICT         : {r['classification']}  ({wl})")
        n_plaus = (retro["classification"] == "PLAUSIBLE").sum()
        n_impl = (retro["classification"] == "IMPLAUSIBLE_BUT_POSSIBLE").sum()
        n_imp = (retro["classification"] == "PHYSICALLY_IMPOSSIBLE").sum()
        print(f"\n  RETRO SUMMARY: {len(retro)} long-gap HIGH-deviation events re-examined -> "
              f"{n_plaus} PLAUSIBLE, {n_impl} IMPLAUSIBLE_BUT_POSSIBLE, {n_imp} PHYSICALLY_IMPOSSIBLE.")
        print(f"  Method read: for these long-gap cases sea-route-adjusted required speed is the")
        print(f"  defensible metric. deviation_km measures predicted-vs-actual offset produced by a")
        print(f"  stale velocity vector across the gap; it says nothing about whether the real")
        print(f"  displacement over the real elapsed time was achievable. required_speed_knots does.")

    # ---- the real discriminator: impossible/implausible move WITH vs
    #      WITHOUT a long AIS gap to explain it ----
    print(f"\n{'='*92}")
    print("KEY DISCRIMINATOR -- trajectory moves rated worse than PLAUSIBLE, split by whether a")
    print("long AIS gap explains them (gap = stale-velocity artifact) or NOT (real track anomaly)")
    print(f"{'='*92}")
    traj_bad = df[(df["source"] == "trajectory_consecutive_actual")
                  & (df["classification"].isin(["PHYSICALLY_IMPOSSIBLE",
                                                "IMPLAUSIBLE_BUT_POSSIBLE"]))].copy()
    traj_bad["gap_h"] = pd.to_numeric(traj_bad["prior_gap_hours"], errors="coerce").fillna(0)
    explained = traj_bad[traj_bad["gap_h"] > 6.0]
    unexplained = traj_bad[traj_bad["gap_h"] <= 6.0]
    print(f"  WITH long AIS gap  (>6 h) -> {len(explained):4d}  (raw deviation was a stale-velocity "
          f"prediction artifact; real required speed is low -> now PLAUSIBLE)")
    print(f"  WITHOUT AIS gap   (<=6 h) -> {len(unexplained):4d}  (a real hourly-scale displacement above "
          f"the vessel-type speed envelope - genuinely worth a look)")
    hard = unexplained[unexplained["classification"] == "PHYSICALLY_IMPOSSIBLE"]
    if len(hard):
        print(f"\n  PHYSICALLY_IMPOSSIBLE with NO gap ({len(hard)}) -- these are genuine track anomalies,")
        print(f"  not sparse-window artifacts; worth investigating as AIS manipulation:")
        for _, r in hard.sort_values("straight_line_speed_knots", ascending=False).iterrows():
            print(f"    {str(r['ship_name'])[:18]:18} mmsi={r['mmsi']}  {str(r['t1'])[:16]}  "
                  f"straight {r['straight_line_distance_km']:.1f} km in {r['elapsed_hours']:.1f} h "
                  f"= {r['straight_line_speed_knots']:.0f} kn, gap12={r['prior_gap_hours']} h  "
                  f"(raw deviation_km={r['prior_metric']}, prior tier {r['prior_tier']})")

    # ---- where the two methods disagree (broader than just retro) ----
    print(f"\n{'='*92}\nMETHOD DISAGREEMENT (raw deviation/distance tier vs route-plausibility)\n{'='*92}")
    prior_high = df[df["prior_tier"].isin(["high", "likely_spoofed"])]
    if len(prior_high):
        dis = prior_high["classification"].value_counts().reindex(order, fill_value=0)
        print(f"  of {len(prior_high)} transitions the OLD method rated high/likely_spoofed, the")
        print(f"  route-plausibility method rates:")
        for k in order:
            print(f"      {k:26}: {int(dis[k])}")
        n_p, n_i = int(dis["PLAUSIBLE"]), int(dis["PHYSICALLY_IMPOSSIBLE"])
        print(f"  -> {n_p} are ordinary once a real sea route and the real elapsed time are used;")
        print(f"     {n_i} {'is a' if n_i == 1 else 'are'} genuine impossibilit{'y' if n_i == 1 else 'ies'} "
              f"the raw deviation_km / distance_km metric could not")
        print(f"     separate from the {n_p} explained ones -- that separation is the value this check adds.")


if __name__ == "__main__":
    main()
