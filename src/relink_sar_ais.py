"""Deterministic SAR<->AIS re-linking via the Hungarian assignment algorithm.

Replaces blind trust in GFW's given MMSI with our own verify-then-repair
re-assignment: every GFW link is checked, only the ones that fail are re-solved
(optimally, one-to-one), and a (re)link is asserted only at matched-tier
confidence -- otherwise the detection is left dark.

Why the Hungarian algorithm and NOT machine learning
---------------------------------------------------
Same reasoning as every other modelling choice here (trajectory_predict's "no
deep learning / no Kalman", spatiotemporal_cluster choosing ST-DBSCAN,
calibrate_thresholds using plain percentiles):
  - fully explainable  -- the assignment is the arg-min of a cost written out
    in full below; every link traces to two numbers (a distance and a
    bracket-check penalty);
  - no training data   -- there is no labelled "correct MMSI" set to learn from;
  - deterministic      -- same input, same output, every run;
  - classical          -- Kuhn-Munkres / Jonker-Volgenant, via
    scipy.optimize.linear_sum_assignment.

Why verify-then-repair and NOT a blind full re-solve
--------------------------------------------------
A blind re-solve of EVERY detection was tested on the India pull (see the
`full` mode and the STEP 5 table): in the New Mangalore fishing grounds there
is almost always some AIS boat a few km from any SAR blob, so an
"assign every detection to its nearest feasible vessel" solve just over-fits
to whichever boat is closest -- it compressed likely_spoofed from ~20 km to
~10 km and "corrected" 79 of 155 GFW links, most of which were fine. GFW's
given MMSI, imperfect as it is, is an actual identity-broadcast correlation
and is right the large majority of the time. So we KEEP GFW's link wherever it
passes the same quality screen calibrate_thresholds.py already uses
(src.sar_ais_quality), and only send the failures + the never-linked
detections into the Hungarian solve.

Reads data/raw/{label}_{sar_detections,ais_positions}.csv. Writes
data/processed/{label}_relinked.csv (one row per SAR detection). Makes NO API
calls -- every candidate vessel is already in the regional AIS pull.

Run:  python -m src.relink_sar_ais                 # India: validation + recalibration
      python -m src.relink_sar_ais <raw_label>
      python -m src.relink_sar_ais <raw_label> full # blind re-solve (diagnostic)
"""
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.match import haversine_km, match_and_classify
from src.interpolate import build_ais_tracks, interpolate_position
from src.sar_ais_quality import bracket_check, screen_identity_pairs, BRACKET_FAR_KM

RAW = "data/raw"
PROC = "data/processed"
DEFAULT_LABEL = "india_mangalore_sep_oct2025"

# ======================================================================
# Re-linking parameters -- justification (house standard: state the
# reasoning, not just the number)
# ======================================================================
#
# SEARCH_RADIUS_KM = 100.0
#   The candidate NET (and the reporting radius), not a match tolerance.
#   Deliberately wide: the known bad GFW links reached 97.5 km (B.ISMAIL), so
#   the search must be wider than that for the algorithm to SEE GFW's wrong
#   assignment as one option among many and get the chance to reject it.
#   Acceptance is governed by LINK_MAX_KM, not by this radius.
SEARCH_RADIUS_KM = 100.0
#
# CANDIDATE_WINDOW_HOURS = 3.0
#   How far either side of the SAR pass to pull a vessel's AIS bins for
#   candidacy and the bracket check. >= the bracket check's own +/-2 h, with
#   margin for the ~3-6 h bin cadence in the India pull.
CANDIDATE_WINDOW_HOURS = 3.0
#
# ---- Cost of assigning SAR detection i to candidate vessel j ----------
#
#     cost(i, j) = d_ij  +  bracket_penalty(i, j)
#
#   d_ij            haversine km between the SAR point and vessel j's
#                   position interpolated (interpolate.py, exactly as
#                   match.py does it) to the SAR timestamp -- a pure "how far
#                   apart" term, and the whole cost when j's own AIS is
#                   consistent with the SAR point.
#   bracket_penalty from src.sar_ais_quality.bracket_check(j, i) -- the SAME
#                   check calibrate_thresholds.py uses:
#                     "consistent"  -> + 0
#                     "unsupported" -> + UNSUPPORTED_PENALTY_KM
#                     "elsewhere"   -> + ELSEWHERE_PENALTY_KM
#   out of SEARCH_RADIUS_KM, or no interpolable track at the SAR time
#                   -> BIG (not a candidate)
#
#   Every pooled detection also gets a private "dark vessel" column at a flat
#   LINK_MAX_KM: choosing it means "no AIS-tracked vessel explains this
#   detection at matched-tier confidence -- leave it dark".
#
# LINK_MAX_KM = 5.0
#   The most a (re)link may cost and still be asserted. We are either
#   OVERRULING GFW's explicit identity correlation or INVENTING a link GFW did
#   not make -- both demand matched-tier confidence, not merely "closer than
#   GFW". 5 km sits just above the calibration's `matched` cutoff (~2-3 km)
#   with room for the ~1.1 km AIS grid and a half-bin time offset. A detection
#   whose best candidate costs more than this is left dark rather than traded
#   from one shaky link to another.
LINK_MAX_KM = 5.0
#
# UNSUPPORTED_PENALTY_KM = 50.0
#   A candidate with no AIS bin within +/-2 h carries no positional evidence
#   (interpolated across a hole). +50 guarantees it loses to the dark column
#   (d >= 0 => cost >= 50 > 5): "we cannot place this vessel, so we will not
#   claim it", matching the calibration screen.
#
# ELSEWHERE_PENALTY_KM = 10_000.0
#   The candidate's OWN AIS brackets the pass yet stays > BRACKET_FAR_KM away
#   throughout -- positive evidence it was not the SAR blob. Effectively a
#   disqualification; a forced assignment at this cost is re-flagged
#   `forced_low_confidence`.
#
# BIG -- hard "not a candidate" sentinel.
LINK_MAX_KM = 5.0
UNSUPPORTED_PENALTY_KM = 50.0
ELSEWHERE_PENALTY_KM = 10_000.0
BIG = 1e7

# Blind-re-solve ("full") mode keeps GFW nowhere and links anything within
# this looser bound -- kept only to demonstrate the dense-traffic over-fit.
FULL_MODE_LINK_MAX_KM = 15.0

_BRACKET_PENALTY = {
    "consistent": 0.0,
    "unsupported": UNSUPPORTED_PENALTY_KM,
    "elsewhere": ELSEWHERE_PENALTY_KM,
}


# ======================================================================
# Data
# ======================================================================
def load_raw(label):
    sar = pd.read_csv(f"{RAW}/{label}_sar_detections.csv", parse_dates=["timestamp"])
    ais = pd.read_csv(f"{RAW}/{label}_ais_positions.csv", parse_dates=["timestamp"])
    return sar, ais


def gfw_links_that_pass_screen(sar, ais):
    """sar_ids whose GFW-given MMSI is a confirmed identity pair that survives
    src.sar_ais_quality.screen_identity_pairs -- these are LOCKED (kept as
    GFW gave them, not re-solved)."""
    m = match_and_classify(sar, ais)
    dist = pd.to_numeric(m["distance_km"], errors="coerce")
    is_id = m["time_gap_seconds"].notna() & (dist >= 0) & dist.notna()
    screen = screen_identity_pairs(m, ais, is_id)
    return set(screen.loc[screen["keep"], "sar_id"].tolist())


# ======================================================================
# One overpass -> optimal assignment of the pooled detections
# ======================================================================
def relink_overpass(sar_grp, ais_df, tracks, locked_ids, link_max_km):
    """All rows of sar_grp share a timestamp. Detections whose sar_id is in
    `locked_ids` keep their GFW MMSI and reserve that vessel; the rest are
    solved by the Hungarian assignment. Returns a list of per-detection dicts.
    """
    ts = pd.Timestamp(sar_grp["timestamp"].iloc[0])
    win = pd.Timedelta(hours=CANDIDATE_WINDOW_HOURS)
    two_h = pd.Timedelta(hours=2)
    nearby = ais_df[(ais_df["timestamp"] >= ts - win) & (ais_df["timestamp"] <= ts + win)]

    det = sar_grp.reset_index(drop=True)
    locked_mask = det["sar_id"].isin(locked_ids).to_numpy()
    reserved_mmsi = {int(mm) for mm in det.loc[locked_mask, "mmsi"].dropna()}

    # --- candidate vessels (excluding reserved ones) ---
    cand_mmsi, cand_lat, cand_lon, cand_name, cand_bins = [], [], [], [], []
    for m in pd.unique(nearby["mmsi"].dropna()):
        mi = int(m)
        if mi in reserved_mmsi:
            continue
        tr = tracks.get(m)
        if tr is None:
            continue
        interp = interpolate_position(tr, ts)
        if interp is None:
            continue
        la, lo, _ = interp
        sub = nearby[(nearby["mmsi"] == m)
                     & (nearby["timestamp"] >= ts - two_h)
                     & (nearby["timestamp"] <= ts + two_h)].sort_values("timestamp")
        nm = nearby.loc[nearby["mmsi"] == m, "ship_name"].dropna()
        cand_mmsi.append(mi)
        cand_lat.append(float(la))
        cand_lon.append(float(lo))
        cand_name.append(str(nm.iloc[0]) if len(nm) else "")
        cand_bins.append(sub)

    pool_idx = np.nonzero(~locked_mask)[0]
    n_pool, n_cand = len(pool_idx), len(cand_mmsi)
    slat = det["lat"].to_numpy(dtype=float)
    slon = det["lon"].to_numpy(dtype=float)

    # --- cost matrix over the POOL rows ---
    cost = np.full((n_pool, n_cand + n_pool), BIG, dtype=float)
    cand_d_full = {}   # j -> haversine to every pool det (for candidate counting)
    for j in range(n_cand):
        d = haversine_km(slat[pool_idx], slon[pool_idx], cand_lat[j], cand_lon[j])
        cand_d_full[j] = d
        for k in np.nonzero(d <= SEARCH_RADIUS_KM)[0]:
            status, _ = bracket_check(cand_bins[j], ts, slat[pool_idx[k]],
                                      slon[pool_idx[k]], far_km=BRACKET_FAR_KM)
            cost[k, j] = d[k] + _BRACKET_PENALTY[status]
    for k in range(n_pool):
        cost[k, n_cand + k] = link_max_km

    if n_pool:
        row_ind, col_ind = linear_sum_assignment(cost)
        picked = dict(zip(row_ind.tolist(), col_ind.tolist()))
    else:
        picked = {}

    rows = []
    for i in range(len(det)):
        drow = det.iloc[i]
        gfw_mmsi = int(drow["mmsi"]) if pd.notna(drow["mmsi"]) else None
        gfw_name = (str(drow.get("ship_name")).strip()
                    if pd.notna(drow.get("ship_name")) else None)

        if locked_mask[i]:
            rows.append(dict(
                sar_id=drow["sar_id"], sar_timestamp=ts,
                sar_lat=slat[i], sar_lon=slon[i],
                gfw_mmsi=gfw_mmsi, gfw_ship_name=gfw_name,
                relinked_mmsi=gfw_mmsi, relinked_name=gfw_name,
                relinked_dist_km=None, assignment_cost=None,
                bracket_status="locked", link_status="gfw_locked",
                change_vs_gfw="gfw_confirmed", n_candidates_in_radius=None))
            continue

        k = int(np.nonzero(pool_idx == i)[0][0])
        j = picked.get(k, n_cand + k)
        n_in_radius = int(sum(cand_d_full[jj][k] <= SEARCH_RADIUS_KM for jj in range(n_cand)))
        if j < n_cand:
            m = cand_mmsi[j]
            d_only = float(haversine_km(slat[i], slon[i], cand_lat[j], cand_lon[j]))
            status, _ = bracket_check(cand_bins[j], ts, slat[i], slon[i], far_km=BRACKET_FAR_KM)
            forced = cost[k, j] >= ELSEWHERE_PENALTY_KM
            relinked_mmsi, relinked_name = m, (cand_name[j] or None)
            relinked_dist, assign_cost = d_only, float(cost[k, j])
            link_status = "forced_low_confidence" if forced else "relinked"
            bracket = status
        else:
            relinked_mmsi = relinked_name = None
            relinked_dist = np.nan
            assign_cost = link_max_km
            link_status = "dark_vessel"
            bracket = None

        if relinked_mmsi is None:
            change = "gfw_rejected_to_dark" if gfw_mmsi is not None else "dark_unchanged"
        elif gfw_mmsi is None:
            change = "newly_linked"
        elif relinked_mmsi == gfw_mmsi:
            change = "gfw_confirmed"
        else:
            change = "gfw_corrected"

        rows.append(dict(
            sar_id=drow["sar_id"], sar_timestamp=ts,
            sar_lat=slat[i], sar_lon=slon[i],
            gfw_mmsi=gfw_mmsi, gfw_ship_name=gfw_name,
            relinked_mmsi=relinked_mmsi, relinked_name=relinked_name,
            relinked_dist_km=(round(relinked_dist, 3) if pd.notna(relinked_dist) else None),
            assignment_cost=round(assign_cost, 3),
            bracket_status=bracket, link_status=link_status,
            change_vs_gfw=change, n_candidates_in_radius=n_in_radius))
    return rows


def relink_dataset(label, mode="repair"):
    sar, ais = load_raw(label)
    tracks = build_ais_tracks(ais)
    link_max = LINK_MAX_KM if mode == "repair" else FULL_MODE_LINK_MAX_KM
    locked = gfw_links_that_pass_screen(sar, ais) if mode == "repair" else set()

    t0 = time.perf_counter()
    rows, sizes = [], []
    for _, grp in sar.groupby("timestamp", sort=True):
        sizes.append(len(grp))
        rows.extend(relink_overpass(grp, ais, tracks, locked, link_max))
    elapsed = time.perf_counter() - t0

    df = pd.DataFrame(rows).sort_values("sar_id").reset_index(drop=True)
    meta = dict(n_sar=len(sar), n_ais=len(ais),
                n_overpasses=sar["timestamp"].nunique(),
                largest_overpass=max(sizes) if sizes else 0,
                n_locked=len(locked), elapsed_s=elapsed, mode=mode)
    return df, sar, ais, meta


# ======================================================================
# Step 4 -- validation on the 5 known-bad India tail-outlier pairs
# ======================================================================
KNOWN_CASES = {
    355: dict(name="B.ISMAIL", gfw_bad=419950651,
              why="inspection found no alternative vessel within 37 km -> expect DARK"),
    195: dict(name="SEA BREEZ", gfw_bad=419951419,
              why="own AIS held a coherent track ~80 km away; nearest alt ~30 km -> expect DARK"),
    741: dict(name="VIKRAM", gfw_bad=419950103,
              why="no clean vessel candidate; nearest AIS objects were fishing-net "
                  "transponders ~6 km -> expect DARK (or a near net-marker)"),
    326: dict(name="THAQWA", gfw_bad=419956049, alt=419950111, alt_name="ST LAWRENCE",
              why="ST LAWRENCE was ~3.1 km away but is claimed by a closer detection "
                  "-> expect reassign-to-ST-LAWRENCE OR DARK, never the GFW MMSI"),
    39: dict(name="SH", gfw_bad=419826316, alt=419951269, alt_name="SRI DATTANJANEYA 2",
             why="SRI DATTANJANEYA 2 was ~0.0 km away but a co-located SAR return "
                 "(sar_id 427) has an equal claim -> expect reassign OR DARK, never the GFW MMSI"),
}


def print_validation(df):
    print("=" * 84)
    print("STEP 4 -- VALIDATION on the 5 known-bad GFW links (India tail outliers)")
    print("  pass criterion: the wrong GFW MMSI is REJECTED (reassigned to a genuine")
    print("  close vessel where one is free, else left DARK -- never kept)")
    print("=" * 84)
    by_id = df.set_index("sar_id")
    n_pass = 0
    for sid, exp in KNOWN_CASES.items():
        if sid not in by_id.index:
            print(f"  sar_id {sid} ({exp['name']}): NOT IN DATASET"); continue
        r = by_id.loc[sid]
        is_dark = r["link_status"] == "dark_vessel"
        kept_bad = (r["relinked_mmsi"] == exp["gfw_bad"])
        got = ("DARK" if is_dark else
               f"{r['relinked_name'] or '?'} ({r['relinked_mmsi']}) @ {r['relinked_dist_km']} km "
               f"[{r['link_status']}]")
        ok = (not kept_bad) and (is_dark or r["link_status"] in ("relinked",))
        n_pass += int(ok)
        print(f"\n  [{sid}] {exp['name']}   GFW link -> MMSI {exp['gfw_bad']} (the bad one)")
        print(f"        re-link  -> {got}   cost={r['assignment_cost']}  "
              f"candidates_in_radius={r['n_candidates_in_radius']}")
        print(f"        {'PASS' if ok else 'REVIEW'}: bad GFW MMSI "
              f"{'rejected' if not kept_bad else 'STILL PRESENT'}. {exp['why']}")
    print(f"\n  ---> {n_pass}/{len(KNOWN_CASES)} known-bad links rejected as expected")
    return n_pass


# ======================================================================
# Step 5 -- recalibrate India with the re-linked assignments
# ======================================================================
def _thr(cal):
    d = cal.get("derived_thresholds_km", {}) or {}
    pre = cal.get("pre_screen_derived_thresholds_km") or d
    return dict(
        n_pre=cal.get("n_confirmed_identity_pairs_pre_screen"),
        n_used=cal.get("n_confirmed_identity_pairs"),
        raw_m=pre.get("matched_max"), raw_s=pre.get("likely_spoofed_min"),
        scr_m=d.get("matched_max"), scr_s=d.get("likely_spoofed_min"))


def _relinked_sar(sar, df):
    out = sar.copy()
    mp = df.set_index("sar_id")["relinked_mmsi"]
    out["mmsi"] = out["sar_id"].map(mp)
    out["ais_matched"] = out["mmsi"].notna()
    return out


def print_recalibration(df, sar, ais, meta):
    from src.calibrate_thresholds import calibrate_distance_thresholds

    print("\n" + "=" * 84)
    print("STEP 5 -- RECALIBRATION: GFW-raw vs mis-attribution screen vs Hungarian re-link")
    print("=" * 84)

    g = _thr(calibrate_distance_thresholds(sar, ais))
    r = _thr(calibrate_distance_thresholds(_relinked_sar(sar, df), ais))

    # blind full re-solve, for contrast (cheap-ish; reuses the same machinery)
    df_full, _, _, _ = relink_dataset(DEFAULT_LABEL, mode="full")
    f = _thr(calibrate_distance_thresholds(_relinked_sar(sar, df_full), ais))

    MANUAL_EXCL_SPOOF = 22.16  # inspect_india_tail_outliers.py, dropping just the 5

    print(f"  {'approach':<38}{'pairs':>7}{'matched_km':>12}{'likely_spoofed_km':>19}")
    print("  " + "-" * 74)
    print(f"  {'raw GFW, no screen':<38}{g['n_pre']:>7}{g['raw_m']:>12.2f}{g['raw_s']:>19.2f}")
    print(f"  {'GFW + mis-attribution screen':<38}{g['n_used']:>7}{g['scr_m']:>12.2f}{g['scr_s']:>19.2f}")
    print(f"  {'manual exclusion of the 5 (reference)':<38}{'~150':>7}{'--':>12}{MANUAL_EXCL_SPOOF:>19.2f}")
    print(f"  {'Hungarian re-link (repair) + screen':<38}{r['n_used']:>7}{r['scr_m']:>12.2f}{r['scr_s']:>19.2f}")
    print(f"  {'blind full re-solve + screen (diag)':<38}{f['n_used']:>7}{f['scr_m']:>12.2f}{f['scr_s']:>19.2f}")
    print("  " + "-" * 74)

    ch = df["change_vs_gfw"].value_counts()
    order = ["gfw_confirmed", "gfw_corrected", "gfw_rejected_to_dark",
             "newly_linked", "dark_unchanged"]
    print("\n  Re-link (repair mode) changes vs GFW's given MMSI:")
    for k in order:
        print(f"    {k:<22} {int(ch.get(k, 0)):>5}")
    newly = int(ch.get("newly_linked", 0))
    corr = int(ch.get("gfw_corrected", 0))
    rej = int(ch.get("gfw_rejected_to_dark", 0))
    print(f"\n  -> {newly} detections GFW left with NO MMSI now carry a genuine "
          f"(matched-tier, < {LINK_MAX_KM:g} km) re-assignment.")
    print(f"  -> {corr} GFW MMSIs replaced with a closer vessel; {rej} rejected to dark.")
    print(f"  -> {meta['n_locked']} GFW links passed the screen and were kept as-is (locked).")

    dm = r["scr_s"] - g["scr_s"]
    print(f"\n  Operative cutoff: GFW+screen likely_spoofed {g['scr_s']:.2f} km  ->  "
          f"re-link(repair)+screen {r['scr_s']:.2f} km  ({dm:+.2f} km).")
    if abs(dm) < 3.0:
        print("  Proper (repair) re-linking does NOT move the derived thresholds "
              "materially beyond what the exclusion screen already gave -- it lands at\n"
              "  the same ~20 km. Its added value is better IDENTITY DATA (corrected /\n"
              "  newly-linked detections for downstream steps), not a different threshold.")
    else:
        print("  Repair re-linking shifts the cutoff beyond the screen -- see the table.")
    print(f"\n  Contrast: the blind full re-solve gives likely_spoofed {f['scr_s']:.2f} km "
          f"and 'corrects' most GFW links -- it over-fits to the nearest boat in dense\n"
          f"  traffic and is NOT used; it is shown only to justify the verify-then-repair "
          f"design.")


# ======================================================================
# Step 6 -- performance / cost
# ======================================================================
def print_performance(meta, df):
    print("\n" + "=" * 84)
    print("STEP 6 -- PERFORMANCE / COST")
    print("=" * 84)
    n = meta["n_sar"]
    per = 1000.0 * meta["elapsed_s"] / n if n else 0.0
    med = int(df["n_candidates_in_radius"].dropna().median()) if df["n_candidates_in_radius"].notna().any() else 0
    mx = int(df["n_candidates_in_radius"].dropna().max()) if df["n_candidates_in_radius"].notna().any() else 0
    print(f"  SAR detections re-linked      : {n}")
    print(f"  AIS rows in the pull          : {meta['n_ais']}")
    print(f"  overpasses (Hungarian solves) : {meta['n_overpasses']}  "
          f"(largest {meta['largest_overpass']} detections)")
    print(f"  GFW links kept as-is (locked) : {meta['n_locked']}  "
          f"(only the rest enter a solve)")
    print(f"  wall clock                    : {meta['elapsed_s']:.1f} s  "
          f"({per:.1f} ms / detection)")
    print(f"  EXTRA API CALLS               : 0  -- every candidate vessel is already "
          f"in the regional AIS pull; re-linking is pure local compute")
    print(f"  added compute per detection   : one AIS time-slice filter + ~N "
          f"interpolations + N bracket checks (N candidates in radius: median {med}, "
          f"max {mx}), then one O(rows*cols) assignment solve per overpass")
    ok = meta["elapsed_s"] < 180 and per < 500
    print(f"  verdict                       : "
          + ("FAST ENOUGH as a standard pipeline step for a single-region run "
             if ok else
             "keep as an OCCASIONAL diagnostic / audit tool ")
          + "(linear in detections; the per-overpass solve is the only super-linear\n"
          "                                  term and stays bounded -- an overpass has "
          "a few hundred detections at most)")


# ======================================================================
def main(argv):
    label = argv[0] if argv and not argv[0] == "full" else DEFAULT_LABEL
    mode = "full" if "full" in argv else "repair"

    df, sar, ais, meta = relink_dataset(label, mode=mode)
    out_path = (f"{PROC}/india_mangalore_relinked.csv" if label == DEFAULT_LABEL
                else f"{PROC}/{label}_relinked.csv")
    df.to_csv(out_path, index=False)

    print_validation(df)
    if label == DEFAULT_LABEL and mode == "repair":
        print_recalibration(df, sar, ais, meta)
    print_performance(meta, df)
    print(f"\n  per-detection re-linking table ({mode} mode): {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
