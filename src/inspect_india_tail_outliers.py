"""Diagnostic: inspect the 5 tail-outlier confirmed pairs from the India
calibration run.

Background
---------
src/calibrate_thresholds.py india_mangalore derives likely_spoofed_min from
the 99th percentile of the CONFIRMED same-vessel (identity-matched) SAR-vs-AIS
distance distribution. On the India baseline that p99 is 78.78 km, versus a
p95 of ~20 km -- the calibration's own `likely_spoofed_tail_check` flagged a
HEAVY TAIL: exactly 5 identity pairs sit beyond the Tukey far-out fence
(Q3 + 3*IQR = 27.48 km) and those 5 alone set the cutoff.

A confirmed identity pair means GFW attributed the SAR detection to an MMSI,
and match.py then interpolated THAT vessel's own AIS track to the SAR
timestamp. A 30-97 km separation with a near-zero interpolation time gap is
therefore either:
  (a) GENUINE_DISPLACEMENT -- the vessel really was ~tens of km from where its
      AIS said, i.e. a real position-spoofing / displacement event that
      happened to fall inside the "clean" baseline window; or
  (b) LIKELY_MISATTRIBUTION -- GFW paired the SAR blob with the wrong MMSI
      (its SAR<->AIS correlator is not perfect in dense coastal traffic), so
      the "pair" is spurious and should not count toward the distribution.

This script decides (a) vs (b) for each of the 5, the same way the earlier
Qatar / Hormuz sanity-check maps did it: pull each vessel's wider +/-2 h AIS
track, look for a closer alternative vessel GFW should have matched instead,
and render one toggleable folium layer per pair.

It is READ-ONLY with respect to data/processed/threshold_calibration_*.json --
it prints a recommendation for the eventual manual_reference decision but does
not touch the calibration output. Its only side effect is writing the map to
data/processed/india_tail_outlier_inspection.html.

Run:  python -m src.inspect_india_tail_outliers

STATUS: this was the one-off diagnostic that VALIDATED the two checks (5/5 on
the India tail). Those checks now live in src/sar_ais_quality.py and run
automatically inside src/calibrate_thresholds.py on every calibration. This
script's own copy of the bracket / alternative-vessel logic is kept as-is as
the historical record of that validation; src.sar_ais_quality is the
production home -- change the logic there, not here.
"""
import numpy as np
import pandas as pd

from src.match import haversine_km, match_and_classify

RAW = "data/raw"
PROC = "data/processed"
RAW_LABEL = "india_mangalore_sep_oct2025"
MAP_OUT = f"{PROC}/india_tail_outlier_inspection.html"

# Mirror calibrate_thresholds.py exactly so the outlier set is identical.
LIKELY_SPOOFED_FALSE_ALARM_RATE = 0.01
LIKELY_SPOOFED_PERCENTILE = 100.0 * (1.0 - LIKELY_SPOOFED_FALSE_ALARM_RATE)

# --- verdict parameters --------------------------------------------------
# TRACK_WINDOW_HOURS: how much of each vessel's own AIS track to pull around
#   the SAR pass -- +/-2 h matches the Qatar/Hormuz visual sanity checks.
TRACK_WINDOW_HOURS = 2.0
#
# ALT_MATCH_KM: a DIFFERENT MMSI whose AIS position is within this range of
#   the SAR point at the pass time is a plausible "true" match. 5 km is
#   generous relative to the ~1.1 km AIS presence grid and comfortably inside
#   even the derived India `discrepant` band, so anything this close is a
#   better explanation for the SAR blob than a 30-97 km jump by the attributed
#   vessel.
ALT_MATCH_KM = 5.0
#
# ALT_MUCH_CLOSER_FRAC: and it must be dramatically closer than the attributed
#   vessel's own interpolated position -- at most this fraction of that
#   distance -- before we call it the real match.
ALT_MUCH_CLOSER_FRAC = 0.5
#
# PLAUSIBLE_SPEED_KN / STRUCT_WRONG_FACTOR: structural test on the track. From
#   the nearest in-window AIS vertex of the attributed vessel, the furthest it
#   could plausibly be by the SAR time is PLAUSIBLE_SPEED_KN * dt. If the SAR
#   point is more than STRUCT_WRONG_FACTOR times that envelope away (and past
#   an absolute floor), the SAR blob sits nowhere near any sane extension of
#   this vessel's own track. 30 kn is well above these coasters'/tankers'
#   service speed; factor 3 keeps the test conservative (only gross geometry
#   failures trip it).
PLAUSIBLE_SPEED_KN = 30.0
STRUCT_WRONG_FACTOR = 3.0
STRUCT_WRONG_FLOOR_KM = 15.0
#
# NO_TRACK_IS_UNSUPPORTED: if the attributed vessel has NO AIS sample at all
#   within +/-2 h of the SAR pass, its position at that time is unconstrained
#   by its own data (match.py interpolated across a multi-hour hole). Treat
#   that as "track does not support the SAR point".
NO_TRACK_IS_UNSUPPORTED = True

LAYER_COLORS = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]


# ======================================================================
# 1. Identify the exact 5 tail-outlier confirmed pairs
# ======================================================================
def load_data():
    sar = pd.read_csv(f"{RAW}/{RAW_LABEL}_sar_detections.csv", parse_dates=["timestamp"])
    ais = pd.read_csv(f"{RAW}/{RAW_LABEL}_ais_positions.csv", parse_dates=["timestamp"])
    return sar, ais


def identity_pairs(sar, ais):
    m = match_and_classify(sar, ais)
    m["distance_km"] = pd.to_numeric(m["distance_km"], errors="coerce")
    idm = m[m["time_gap_seconds"].notna() & m["distance_km"].notna() & (m["distance_km"] >= 0)].copy()
    return m, idm


def tail_outliers(idm):
    d = idm["distance_km"].to_numpy(dtype=float)
    q1, q3 = np.percentile(d, [25, 75])
    fence = float(q3 + 3.0 * (q3 - q1))
    out = idm[idm["distance_km"] > fence].sort_values("distance_km", ascending=False).copy()
    return out, fence


# ======================================================================
# 2. Wider +/-2 h own-vessel track
# ======================================================================
def vessel_track(ais, mmsi, sar_ts, hours=TRACK_WINDOW_HOURS):
    lo = sar_ts - pd.Timedelta(hours=hours)
    hi = sar_ts + pd.Timedelta(hours=hours)
    t = ais[(ais["mmsi"] == mmsi) & (ais["timestamp"] >= lo) & (ais["timestamp"] <= hi)]
    return t.sort_values("timestamp").reset_index(drop=True)


# ======================================================================
# 4. Alternative-match search: any OTHER MMSI near the SAR point at pass time
# ======================================================================
def alternative_matches(ais, sar_row, exclude_mmsi, hours=1.0):
    """Every AIS position from a different MMSI within +/-`hours` of the SAR
    pass, ranked by distance to the SAR point. Returns a DataFrame."""
    sar_ts = sar_row["timestamp"]
    lo, hi = sar_ts - pd.Timedelta(hours=hours), sar_ts + pd.Timedelta(hours=hours)
    c = ais[(ais["timestamp"] >= lo) & (ais["timestamp"] <= hi) & (ais["mmsi"] != exclude_mmsi)].copy()
    if c.empty:
        return c
    c["dist_km"] = haversine_km(sar_row["lat"], sar_row["lon"],
                                c["lat"].to_numpy(dtype=float), c["lon"].to_numpy(dtype=float))
    c["dt_min"] = (c["timestamp"] - sar_ts).dt.total_seconds().abs() / 60.0
    # nearest position per candidate MMSI
    c = c.sort_values("dist_km").drop_duplicates("mmsi", keep="first")
    return c.sort_values("dist_km").reset_index(drop=True)


# ======================================================================
# 3 + 5. Structural check + per-pair verdict
# ======================================================================
def assess_pair(pair, track, alts):
    sar_lat, sar_lon = float(pair["sar_lat"]), float(pair["sar_lon"])
    sar_ts = pair["sar_timestamp"] if "sar_timestamp" in pair else pair["timestamp"]
    sar_ts = pd.Timestamp(sar_ts)
    gap_h = float(pair["time_gap_seconds"]) / 3600.0

    info = {
        "n_track_pts_pm2h": int(len(track)),
        "nearest_own_vertex_km": None,
        "dt_to_nearest_vertex_h": None,
        "plausible_reach_km": None,
        "struct_wrong": None,
        "best_alt_mmsi": None,
        "best_alt_name": None,
        "best_alt_km": None,
        "best_alt_dt_min": None,
        "better_alt_exists": False,
    }

    # structural test against the vessel's own +/-2 h track
    if len(track) == 0:
        info["struct_wrong"] = bool(NO_TRACK_IS_UNSUPPORTED)
        info["struct_reason"] = "no own AIS sample within +/-2 h of the SAR pass"
    else:
        dvx = haversine_km(sar_lat, sar_lon,
                           track["lat"].to_numpy(dtype=float),
                           track["lon"].to_numpy(dtype=float))
        j = int(np.argmin(dvx))
        info["nearest_own_vertex_km"] = round(float(dvx[j]), 3)
        dt_h = abs((track.iloc[j]["timestamp"] - sar_ts).total_seconds()) / 3600.0
        info["dt_to_nearest_vertex_h"] = round(dt_h, 3)
        reach = PLAUSIBLE_SPEED_KN * 1.852 * max(dt_h, gap_h, 1e-6)
        info["plausible_reach_km"] = round(reach, 3)

        # (a) reach test: SAR point beyond any sane extrapolation from the
        #     nearest own vertex.
        reach_wrong = dvx[j] > STRUCT_WRONG_FACTOR * reach and dvx[j] > STRUCT_WRONG_FLOOR_KM
        # (b) bracket test (stronger): the vessel's own AIS has a bin BOTH
        #     before and after the SAR pass, and the CLOSEST of those still
        #     leaves the SAR point far away -> the track coherently places the
        #     vessel elsewhere across the whole pass, so the SAR blob is not
        #     it. This is what catches a vessel steaming a tight coherent line
        #     ~80 km from the SAR point (the reach test alone lets that
        #     through because 80 km is "reachable" at 30 kn over ~2 h).
        ts = track["timestamp"]
        brackets = bool((ts < sar_ts).any() and (ts > sar_ts).any())
        bracket_wrong = brackets and dvx[j] > STRUCT_WRONG_FLOOR_KM
        info["struct_wrong"] = bool(reach_wrong or bracket_wrong)
        if bracket_wrong:
            info["struct_reason"] = (
                f"own AIS brackets the SAR pass (bin before AND after) yet "
                f"stays >= {dvx[j]:.1f} km away throughout -- vessel was "
                f"coherently elsewhere"
            )
        elif reach_wrong:
            info["struct_reason"] = (
                f"SAR point {dvx[j]:.1f} km from nearest own track vertex; "
                f"plausible reach only ~{reach:.1f} km"
            )
        else:
            info["struct_reason"] = (
                f"SAR point {dvx[j]:.1f} km from nearest own track vertex; "
                f"within plausible reach ~{reach:.1f} km"
            )

    # closer alternative vessel?
    if alts is not None and not alts.empty:
        b = alts.iloc[0]
        info["best_alt_mmsi"] = int(b["mmsi"]) if pd.notna(b["mmsi"]) else None
        info["best_alt_name"] = (b.get("ship_name") or "") if pd.notna(b.get("ship_name")) else ""
        info["best_alt_km"] = round(float(b["dist_km"]), 3)
        info["best_alt_dt_min"] = round(float(b["dt_min"]), 1)
        info["better_alt_exists"] = bool(
            b["dist_km"] <= ALT_MATCH_KM
            and b["dist_km"] <= ALT_MUCH_CLOSER_FRAC * float(pair["distance_km"])
        )

    if info["better_alt_exists"] or info["struct_wrong"]:
        verdict = "LIKELY_MISATTRIBUTION"
    else:
        verdict = "GENUINE_DISPLACEMENT"
    info["verdict"] = verdict
    return info


# ======================================================================
# 3. Folium map, one toggleable layer per pair
# ======================================================================
def build_map(pairs, tracks, alts_list, assessments):
    import folium

    center = [float(pairs["sar_lat"].mean()), float(pairs["sar_lon"].mean())]
    m = folium.Map(location=center, zoom_start=8, tiles="OpenStreetMap")

    for i, (_, pair) in enumerate(pairs.iterrows()):
        color = LAYER_COLORS[i % len(LAYER_COLORS)]
        mmsi = int(pair["matched_mmsi"])
        name = (pair.get("ship_name") or "").strip() or "?"
        a = assessments[i]
        fg = folium.FeatureGroup(
            name=f"{i+1}. {name} ({mmsi})  {pair['distance_km']:.0f} km  -> {a['verdict']}",
            show=(i == 0),
        )

        # SAR detection point
        folium.CircleMarker(
            [float(pair["sar_lat"]), float(pair["sar_lon"])],
            radius=7, color="black", weight=2, fill=True, fill_color=color, fill_opacity=0.95,
            popup=folium.Popup(
                f"<b>SAR detection</b> (sar_id {pair['sar_id']})<br>"
                f"{pair['sar_timestamp']}<br>"
                f"{float(pair['sar_lat']):.4f}, {float(pair['sar_lon']):.4f}<br>"
                f"attributed MMSI: {mmsi} ({name})<br>"
                f"distance to attributed AIS: {pair['distance_km']:.2f} km<br>"
                f"interp time gap: {float(pair['time_gap_seconds'])/3600:.2f} h",
                max_width=340),
        ).add_to(fg)

        # attributed / interpolated AIS point
        if pd.notna(pair["ais_lat"]) and pd.notna(pair["ais_lon"]):
            folium.CircleMarker(
                [float(pair["ais_lat"]), float(pair["ais_lon"])],
                radius=5, color=color, weight=2, fill=True, fill_color="white", fill_opacity=0.9,
                popup=f"attributed AIS position (interpolated to SAR time) for MMSI {mmsi}",
            ).add_to(fg)
            folium.PolyLine(
                [[float(pair["sar_lat"]), float(pair["sar_lon"])],
                 [float(pair["ais_lat"]), float(pair["ais_lon"])]],
                color=color, weight=2, dash_array="6", opacity=0.8,
                popup=f"{pair['distance_km']:.1f} km SAR<->attributed-AIS",
            ).add_to(fg)

        # the vessel's own +/-2 h AIS track
        tr = tracks[i]
        if len(tr) >= 1:
            if len(tr) >= 2:
                folium.PolyLine(
                    tr[["lat", "lon"]].values.tolist(),
                    color=color, weight=3, opacity=0.9,
                    popup=f"MMSI {mmsi} AIS track +/-2 h ({len(tr)} bins)",
                ).add_to(fg)
            for _, r in tr.iterrows():
                folium.CircleMarker(
                    [float(r["lat"]), float(r["lon"])],
                    radius=3, color=color, fill=True, fill_opacity=0.7,
                    popup=f"{r['timestamp']}  ({float(r['lat']):.4f}, {float(r['lon']):.4f})",
                ).add_to(fg)

        # alternative candidate vessels near the SAR point at pass time
        alts = alts_list[i]
        if alts is not None and not alts.empty:
            for _, r in alts.head(6).iterrows():
                folium.CircleMarker(
                    [float(r["lat"]), float(r["lon"])],
                    radius=4, color="#555555", weight=1, fill=True, fill_color="yellow",
                    fill_opacity=0.85,
                    popup=(f"ALT candidate MMSI {int(r['mmsi'])} "
                           f"{(r.get('ship_name') or '')}<br>"
                           f"{r['dist_km']:.2f} km from SAR point, "
                           f"dt {r['dt_min']:.0f} min"),
                ).add_to(fg)

        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(MAP_OUT)
    return MAP_OUT


# ======================================================================
# Driver
# ======================================================================
def main():
    sar, ais = load_data()
    full_match, idm = identity_pairs(sar, ais)
    out, fence = tail_outliers(idm)

    print("=" * 78)
    print("INDIA CALIBRATION -- TAIL-OUTLIER CONFIRMED PAIRS")
    print("=" * 78)
    print(f"confirmed identity pairs: {len(idm)}   "
          f"Tukey far-out fence (Q3 + 3*IQR): {fence:.2f} km")
    print(f"pairs beyond the fence:   {len(out)}\n")

    tracks, alts_list, assessments = [], [], []
    for i, (_, pair) in enumerate(out.iterrows(), start=1):
        mmsi = int(pair["matched_mmsi"])
        name = (pair.get("ship_name") or "").strip() or "?"
        sar_ts = pd.Timestamp(pair["sar_timestamp"])
        gap_h = float(pair["time_gap_seconds"]) / 3600.0

        tr = vessel_track(ais, mmsi, sar_ts)
        sar_row = {"timestamp": sar_ts, "lat": float(pair["sar_lat"]), "lon": float(pair["sar_lon"])}
        alts = alternative_matches(ais, sar_row, exclude_mmsi=mmsi, hours=1.0)
        a = assess_pair(pair, tr, alts)

        tracks.append(tr)
        alts_list.append(alts)
        assessments.append(a)

        print("-" * 78)
        print(f"[{i}] {name}  (MMSI {mmsi})   sar_id {pair['sar_id']}")
        print(f"    SAR detection : {sar_ts}   ({float(pair['sar_lat']):.4f}, {float(pair['sar_lon']):.4f})")
        print(f"    attributed AIS: interpolated to SAR time   "
              f"({float(pair['ais_lat']):.4f}, {float(pair['ais_lon']):.4f})")
        print(f"    distance      : {pair['distance_km']:.2f} km      "
              f"interp time gap: {gap_h:.2f} h")
        print(f"    own +/-2h track: {a['n_track_pts_pm2h']} AIS bin(s)")
        if a["n_track_pts_pm2h"]:
            print(f"      nearest own track vertex to SAR point: "
                  f"{a['nearest_own_vertex_km']} km  "
                  f"(dt {a['dt_to_nearest_vertex_h']} h; plausible reach "
                  f"~{a['plausible_reach_km']} km)")
        print(f"      structural check: "
              f"{'SAR POINT OFF-TRACK' if a['struct_wrong'] else 'consistent with track'}"
              f"   [{a.get('struct_reason','')}]")
        if a["best_alt_km"] is not None:
            tag = "  <-- better match" if a["better_alt_exists"] else ""
            print(f"    closest alternative vessel at pass time: "
                  f"MMSI {a['best_alt_mmsi']} {a['best_alt_name']}  "
                  f"{a['best_alt_km']} km  (dt {a['best_alt_dt_min']} min){tag}")
            if alts is not None and len(alts) > 1:
                nxt = alts.iloc[1]
                print(f"      next closest: MMSI {int(nxt['mmsi'])} "
                      f"{(nxt.get('ship_name') or '')}  {nxt['dist_km']:.2f} km")
        else:
            print("    no alternative AIS vessel within +/-1 h anywhere in the bbox")
        print(f"    VERDICT: {a['verdict']}")

    # --------------------------------------------------------------
    # 6. Practical conclusion: recompute likely_spoofed without the
    #    misattribution pairs
    # --------------------------------------------------------------
    mis_idx = [i for i, a in enumerate(assessments) if a["verdict"] == "LIKELY_MISATTRIBUTION"]
    gen_idx = [i for i, a in enumerate(assessments) if a["verdict"] == "GENUINE_DISPLACEMENT"]
    mis_sar_ids = set(out.iloc[mis_idx]["sar_id"].tolist())

    d_all = idm["distance_km"].to_numpy(dtype=float)
    keep = idm[~idm["sar_id"].isin(mis_sar_ids)]["distance_km"].to_numpy(dtype=float)

    p95_all = float(np.percentile(d_all, 95))
    p99_all = float(np.percentile(d_all, LIKELY_SPOOFED_PERCENTILE))
    p95_keep = float(np.percentile(keep, 95))
    p99_keep = float(np.percentile(keep, LIKELY_SPOOFED_PERCENTILE))

    print("\n" + "=" * 78)
    print("PRACTICAL CONCLUSION")
    print("=" * 78)
    print(f"  of 5 tail outliers:  {len(mis_idx)} LIKELY_MISATTRIBUTION, "
          f"{len(gen_idx)} GENUINE_DISPLACEMENT")
    print(f"  current likely_spoofed_min (p99 of all {len(idm)} pairs) : {p99_all:.2f} km")
    print(f"  current p95                                             : {p95_all:.2f} km")
    if mis_idx:
        print(f"  dropping the {len(mis_idx)} misattribution pair(s) "
              f"-> {len(keep)} pairs:")
        print(f"    recomputed p99 (likely_spoofed_min) : {p99_keep:.2f} km")
        print(f"    recomputed p95                      : {p95_keep:.2f} km")
    print()

    if len(mis_idx) == 5:
        print("  ALL 5 are misattribution artifacts. RECOMMENDATION: recompute the")
        print(f"  India likely_spoofed_min with these 5 sar_ids excluded -> it lands at")
        print(f"  ~{p99_keep:.0f} km, in line with the p95 (~{p95_keep:.0f} km) and close to the Gulf's")
        print("  hand-set 20 km. The heavy tail was a data-quality artifact of GFW's")
        print("  SAR<->AIS correlator, not evidence of real spoofing in the baseline.")
        print(f"  Exclude sar_ids {sorted(mis_sar_ids)} in a follow-up calibrate_thresholds")
        print("  pass (or add an identity-pair sanity filter there) before setting")
        print("  manual_reference for india_mangalore.")
    elif len(mis_idx) >= 3:
        print(f"  MOST ({len(mis_idx)}/5) are misattribution. RECOMMENDATION: exclude those and")
        print(f"  recompute -> likely_spoofed_min ~{p99_keep:.0f} km. Re-inspect the "
              f"{len(gen_idx)} pair(s)")
        print("  flagged GENUINE before deciding whether to keep them; if they survive a")
        print("  closer look they raise the India cutoff modestly above the Gulf's 20 km.")
    elif len(gen_idx) >= 3:
        print(f"  MOST ({len(gen_idx)}/5) look like GENUINE displacement. That means the India")
        print("  baseline is NOT clean of position anomalies and likely_spoofed_min should")
        print(f"  stay near the p99 (~{p99_all:.0f} km) -- a lower cutoff would fire on events")
        print("  that really are ~tens of km displacements. Investigate those vessels")
        print("  individually and confirm the Sept-Oct window before trusting either value.")
    else:
        print(f"  MIXED ({len(mis_idx)} misattribution / {len(gen_idx)} genuine). Exclude the")
        print(f"  misattribution pairs (-> ~{p99_keep:.0f} km) and treat the genuine one(s) as")
        print("  real baseline anomalies to investigate; do not finalise manual_reference")
        print("  until each genuine pair is explained.")

    map_path = build_map(out.reset_index(drop=True), tracks, alts_list, assessments)
    print(f"\n  map ({len(out)} toggleable layers): {map_path}")
    print("  (calibration JSON left untouched -- this is a diagnostic only)")


if __name__ == "__main__":
    main()
