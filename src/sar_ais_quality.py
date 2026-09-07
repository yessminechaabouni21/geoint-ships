"""SAR<->AIS identity-pair quality screening -- catches GFW SAR-to-MMSI
mis-attribution.

Why this module exists
----------------------
GFW's SAR detection product ships some detections pre-matched to an MMSI, and
match.py's identity path trusts that attribution (it interpolates THAT
vessel's own AIS track to the SAR time and treats the result as a confirmed
same-vessel pair). On the India / New Mangalore Sept-Oct 2025 pull that trust
was misplaced: of the confirmed identity pairs, the 5 with the largest
SAR-vs-own-AIS distance (30-97 km) were ALL GFW mis-attribution, not real
position displacement -- verified vessel-by-vessel in
src/inspect_india_tail_outliers.py (5 of 5). On one SAR overpass only 21 of
637 detections got an MMSI at all, and several of those 21 were 30-100 km
wrong.

That is a GENERAL data-quality problem in dense coastal traffic, not an
India-specific one. GFW's SAR<->AIS correlator degrades wherever many vessels
sit close together with patchy AIS, which describes most of the coastal AOIs
this project will move to next. So this screen is designed to run in TWO
places:
  1. threshold calibration (src/calibrate_thresholds.py) -- so mis-attributed
     pairs never enter the distribution the matched / likely_spoofed cutoffs
     are derived from (a single 97 km bad pair moved India's likely_spoofed
     from ~22 km to ~79 km);
  2. real detection runs, later -- so a mis-attributed pair is not reported as
     a spoofing hit.

The two checks (both lifted verbatim from the validated India tail inspection)
--------------------------------------------------------------------------
1. `bracket_check` -- does the attributed vessel's OWN AIS have a bin both
   BEFORE and AFTER the SAR pass, with the nearest of those bins still far
   (> BRACKET_FAR_KM) from the SAR point? Then the vessel was coherently
   somewhere else across the whole pass and this SAR blob is not it.

   This REPLACES the earlier, weaker "could the vessel have physically
   reached this point at some plausible speed" test. That test passed
   SEA BREEZ on the first India inspection pass -- a vessel holding a tight
   coherent track ~80 km from the SAR point, which is "reachable" at 30 kn
   over ~2 h but is plainly not the same object. Bracketing AIS on both
   sides, staying far throughout, is not something a genuine same-vessel
   match ever does.

   A vessel with NO own AIS bin inside the +/-window is "unsupported": its
   attributed position was interpolated across a multi-hour hole (THAQWA had
   a 76 h gap) and is not constrained by the vessel's own data.

2. `alternative_vessel_check` -- is there a DIFFERENT vessel whose AIS
   position sits much closer to the SAR point (<= ALT_MATCH_KM, and <=
   ALT_CLOSER_FRAC of the attributed pair's distance) at the pass time? Then
   GFW should have matched the SAR blob to THAT vessel.

A pair failing EITHER check is `suspected_misattribution` and is dropped from
the calibration population (and should be dropped / flagged in detection).

Known limitation
----------------
`bracket_check` alone cannot separate mis-attribution from a genuine
AIS-position-spoofing event: in both, the attributed vessel's real track is
elsewhere. `alternative_vessel_check` resolves the common mis-attribution case
(a closer real vessel exists, or the SAR blob is a genuinely dark craft that
was spuriously tagged with a far-away MMSI). For CALIBRATION this is the safe
direction to err -- the baseline window is meant to be incident-free, so a
handful of large-distance identity pairs are contamination to remove
regardless of which mechanism produced them. For DETECTION, a pair that fails
the bracket check but has NO better alternative should be surfaced for review
(could be real spoofing), not silently dropped -- callers get the per-pair
detail to make that call.
"""
import numpy as np
import pandas as pd

from src.match import haversine_km

# --- screening parameters (carried over verbatim from the validated
#     src/inspect_india_tail_outliers.py run; see this module's docstring
#     for the reasoning behind each) -----------------------------------
TRACK_WINDOW_HOURS = 2.0   # how much of the attributed vessel's own track to
#                            consider around the SAR pass (same +/-2 h used by
#                            the Qatar / Hormuz visual sanity-check maps)
BRACKET_FAR_KM = 15.0      # "far" for the bracket check. Well above the
#                            ~1.1 km AIS presence grid and above any
#                            legitimate match noise (Gulf confirmed-pair p95
#                            was ~16 km), so a real same-vessel pair does not
#                            trip it; a vessel whose own AIS sits > 15 km from
#                            the SAR point on BOTH sides of the pass was
#                            demonstrably elsewhere.
ALT_WINDOW_HOURS = 1.0     # time window for the alternative-vessel search
ALT_MATCH_KM = 5.0         # a different vessel this close to the SAR point is
#                            a plausible true match (comfortably inside even a
#                            widened `discrepant` band)
ALT_CLOSER_FRAC = 0.5      # ...and it must be <= half the attributed pair's
#                            distance before we call it clearly better

# SCREEN_MIN_DISTANCE_KM: only screen pairs whose attributed SAR-vs-own-AIS
#   distance exceeds this. Below it a mis-attribution cannot distort the
#   calibration -- such a pair is at worst `discrepant`, sits near the median,
#   and neither moves `matched` (p50 ~= 2-3 km) nor pushes `likely_spoofed`
#   (p99). Screening below this floor is actively harmful in sparse-coverage
#   regions: on the India pull, AIS bins are 3-6 h apart (which is why
#   eps_temporal calibrates to 6 h, not the Gulf's 2 h), so "no own bin within
#   +/-2 h" and "another boat within ~2 km" are ordinary there, not evidence
#   of a bad attribution -- an ungated screen wrongly dropped 61 of 155 pairs,
#   including 0.3-2 km matches. Gated at 10 km it targets exactly the pairs
#   that corrupt the p95-p99 region the thresholds are read from, and it is
#   also the right behaviour for later detection use (a would-be
#   likely_spoofed alert is by definition well past this floor).
SCREEN_MIN_DISTANCE_KM = 10.0

# UNSUPPORTED_ONLY_MIN_KM: a "bracket=unsupported" verdict (no own AIS bin
#   within +/-2 h -- just a coverage gap the interpolation bridged) is a WEAK
#   signal on its own in a sparse-coverage region: many legitimate pairs there
#   have a 2-6 h gap to their nearest bin. So an unsupported pair is only
#   dropped if it ALSO has a clearly-better alternative vessel, OR its
#   distance is extreme (beyond this floor -- i.e. up in the region the manual
#   India inspection actually adjudicated, where a multi-hour interpolation
#   error simply cannot account for the separation). "elsewhere" (own AIS
#   present on BOTH sides and staying far) and "better alternative vessel"
#   remain strong signals and fire from SCREEN_MIN_DISTANCE_KM up.
UNSUPPORTED_ONLY_MIN_KM = 25.0


def own_track_window(ais_df, mmsi, sar_ts, window_hours=TRACK_WINDOW_HOURS):
    """The attributed vessel's own AIS bins within +/- window_hours of the SAR
    pass, sorted by time."""
    lo = sar_ts - pd.Timedelta(hours=window_hours)
    hi = sar_ts + pd.Timedelta(hours=window_hours)
    t = ais_df[(ais_df["mmsi"] == mmsi)
               & (ais_df["timestamp"] >= lo)
               & (ais_df["timestamp"] <= hi)]
    return t.sort_values("timestamp")


def bracket_check(track_window, sar_ts, sar_lat, sar_lon, far_km=BRACKET_FAR_KM):
    """Return (status, detail).

    status:
      "elsewhere"   -- own AIS brackets the SAR pass (a bin before AND after)
                       and the nearest of those bins is still > far_km from
                       the SAR point: vessel coherently elsewhere, SAR blob is
                       not it.
      "unsupported" -- no own AIS bin within the window at all: attributed
                       position interpolated across a multi-hour hole.
      "consistent"  -- own AIS comes within far_km of the SAR point around the
                       pass, or does not bracket it: attribution is physically
                       consistent.
    """
    n = len(track_window)
    if n == 0:
        return "unsupported", {
            "n_own_bins": 0,
            "nearest_own_bin_km": None,
            "brackets_pass": False,
            "reason": f"no own AIS bin within +/-{TRACK_WINDOW_HOURS:g} h of the SAR pass",
        }
    d = haversine_km(sar_lat, sar_lon,
                     track_window["lat"].to_numpy(dtype=float),
                     track_window["lon"].to_numpy(dtype=float))
    nearest = float(np.min(d))
    ts = track_window["timestamp"]
    brackets = bool((ts < sar_ts).any() and (ts > sar_ts).any())
    if brackets and nearest > far_km:
        return "elsewhere", {
            "n_own_bins": n,
            "nearest_own_bin_km": round(nearest, 3),
            "brackets_pass": True,
            "reason": (f"own AIS brackets the SAR pass (bin before AND after) "
                       f"yet stays >= {nearest:.1f} km away throughout"),
        }
    return "consistent", {
        "n_own_bins": n,
        "nearest_own_bin_km": round(nearest, 3),
        "brackets_pass": brackets,
        "reason": f"own AIS within {nearest:.1f} km of the SAR point around the pass",
    }


def alternative_vessel_check(ais_df, sar_ts, sar_lat, sar_lon, attributed_mmsi,
                             attributed_distance_km,
                             window_hours=ALT_WINDOW_HOURS,
                             alt_match_km=ALT_MATCH_KM,
                             closer_frac=ALT_CLOSER_FRAC):
    """Return (better_alt_exists: bool, detail).

    True iff some other MMSI's AIS position within +/- window_hours of the SAR
    pass is <= alt_match_km from the SAR point AND <= closer_frac of the
    attributed pair's distance -- i.e. GFW should plausibly have matched the
    SAR blob to that vessel instead.
    """
    lo = sar_ts - pd.Timedelta(hours=window_hours)
    hi = sar_ts + pd.Timedelta(hours=window_hours)
    c = ais_df[(ais_df["timestamp"] >= lo)
               & (ais_df["timestamp"] <= hi)
               & (ais_df["mmsi"] != attributed_mmsi)].copy()
    if c.empty:
        return False, {"best_alt_mmsi": None, "best_alt_name": None,
                       "best_alt_km": None,
                       "reason": f"no other vessel within +/-{window_hours:g} h"}
    c["dist_km"] = haversine_km(sar_lat, sar_lon,
                                c["lat"].to_numpy(dtype=float),
                                c["lon"].to_numpy(dtype=float))
    b = c.loc[c["dist_km"].idxmin()]
    better = bool(b["dist_km"] <= alt_match_km
                  and b["dist_km"] <= closer_frac * float(attributed_distance_km))
    name = b.get("ship_name")
    return better, {
        "best_alt_mmsi": int(b["mmsi"]) if pd.notna(b["mmsi"]) else None,
        "best_alt_name": (str(name) if pd.notna(name) else None),
        "best_alt_km": round(float(b["dist_km"]), 3),
        "reason": (
            (f"alternative vessel {b['dist_km']:.2f} km from SAR point "
             f"(<= {alt_match_km:g} km and <= {closer_frac:g}x the attributed "
             f"{attributed_distance_km:.1f} km)")
            if better else
            f"closest other vessel {b['dist_km']:.2f} km -- not clearly better"
        ),
    }


def screen_identity_pairs(match_df, ais_df, identity_mask):
    """Run both checks over every confirmed identity pair.

    match_df: output of src.match.match_and_classify (needs sar_id,
      sar_timestamp, sar_lat, sar_lon, matched_mmsi, distance_km,
      time_gap_seconds, ship_name).
    identity_mask: boolean Series over match_df selecting the confirmed
      identity pairs (time_gap_seconds not null, distance_km valid).

    Returns a DataFrame with one row per identity pair: pair fields plus
    bracket_status, better_alt_exists, keep (bool), exclusion_reason.
    A pair is dropped (keep=False, "suspected_misattribution") if the bracket
    check is "elsewhere" or "unsupported", OR a clearly-better alternative
    vessel exists.
    """
    rows = []
    for _, p in match_df[identity_mask].iterrows():
        sar_ts = pd.Timestamp(p["sar_timestamp"])
        mmsi = p["matched_mmsi"]
        sar_lat, sar_lon = float(p["sar_lat"]), float(p["sar_lon"])
        dist = float(p["distance_km"])

        # Gate: pairs at or below SCREEN_MIN_DISTANCE_KM cannot distort the
        # upper-tail calibration and are kept unscreened (running the checks
        # on them just produces false positives in sparse-coverage regions).
        if dist <= SCREEN_MIN_DISTANCE_KM:
            rows.append({
                "sar_id": p["sar_id"],
                "matched_mmsi": int(mmsi) if pd.notna(mmsi) else None,
                "ship_name": (str(p.get("ship_name")).strip()
                              if pd.notna(p.get("ship_name")) else None),
                "sar_timestamp": sar_ts,
                "distance_km": round(dist, 3),
                "time_gap_h": round(float(p["time_gap_seconds"]) / 3600.0, 3),
                "bracket_status": "not_screened",
                "nearest_own_bin_km": None,
                "better_alt_exists": False,
                "best_alt_mmsi": None,
                "best_alt_km": None,
                "keep": True,
                "exclusion_reason": "",
            })
            continue

        tw = own_track_window(ais_df, mmsi, sar_ts)
        b_status, b_det = bracket_check(tw, sar_ts, sar_lat, sar_lon)
        better, a_det = alternative_vessel_check(
            ais_df, sar_ts, sar_lat, sar_lon, mmsi, dist)

        # "elsewhere" is a strong signal at any screened distance; "unsupported"
        # (a bare coverage gap) needs corroboration -- a better alternative or
        # an extreme distance -- before it excludes.
        fail_elsewhere = b_status == "elsewhere"
        fail_unsupported = b_status == "unsupported" and (
            better or dist > UNSUPPORTED_ONLY_MIN_KM
        )
        keep = not (fail_elsewhere or fail_unsupported or better)
        parts = []
        if fail_elsewhere:
            parts.append(f"bracket=elsewhere ({b_det['reason']})")
        if b_status == "unsupported" and not keep:
            parts.append(f"bracket=unsupported ({b_det['reason']})")
        if better:
            parts.append(a_det["reason"])
        reason = "; ".join(parts)

        rows.append({
            "sar_id": p["sar_id"],
            "matched_mmsi": int(mmsi) if pd.notna(mmsi) else None,
            "ship_name": (str(p.get("ship_name")).strip()
                          if pd.notna(p.get("ship_name")) else None),
            "sar_timestamp": sar_ts,
            "distance_km": round(dist, 3),
            "time_gap_h": round(float(p["time_gap_seconds"]) / 3600.0, 3),
            "bracket_status": b_status,
            "nearest_own_bin_km": b_det["nearest_own_bin_km"],
            "better_alt_exists": better,
            "best_alt_mmsi": a_det["best_alt_mmsi"],
            "best_alt_km": a_det["best_alt_km"],
            "keep": keep,
            "exclusion_reason": reason,
        })
    cols = ["sar_id", "matched_mmsi", "ship_name", "sar_timestamp", "distance_km",
            "time_gap_h", "bracket_status", "nearest_own_bin_km",
            "better_alt_exists", "best_alt_mmsi", "best_alt_km", "keep",
            "exclusion_reason"]
    return pd.DataFrame(rows, columns=cols)
