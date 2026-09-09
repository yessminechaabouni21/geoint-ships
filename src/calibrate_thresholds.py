"""Automatic, region-adaptive threshold calibration.

Why this exists
---------------
Every classification cutoff in this project so far was hand-tuned against Gulf
data: match.py's 1.5 km / 20 km SAR-vs-AIS distance tiers, and
spatiotemporal_cluster.py's 8 km / 2 h ST-DBSCAN epsilons. Moving the pipeline
to a new region (India first) otherwise means re-running a control period by
hand and eyeballing whether those Gulf numbers still look sane -- slow,
subjective, and not reproducible.

This module replaces that with the GeoTrackNet principle (Nguyen et al.,
"GeoTrackNet -- A Maritime Anomaly Detector using Probabilistic Neural Network
Representation of AIS Tracks and A Contrario Detection", IEEE T-ITS 2022; and
the 2018 MTAD-GAN precursor): a detection threshold should be a percentile of
the region's OWN statistical distribution of the quantity being thresholded,
measured over a clean (incident-free) baseline -- NOT a physical constant
carried over from somewhere else. A "spoofing" alert then means "this pair is
past the 99.9th percentile of what incident-free traffic in THIS region looks
like", which re-scales automatically when the region's traffic density, lane
geometry, and AIS/SAR coverage change.

For the ST-DBSCAN parameters it follows the adaptive-parameter DBSCAN idea of
Bai et al. ("An Optimized DBSCAN Algorithm with Adaptive Parameters", and the
KANN-DBSCAN family): derive eps from the knee of the region's own
nearest-neighbour distance curve rather than fixing it.

Mis-attribution screen (runs on EVERY calibration)
------------------------------------------------
Before any percentile is taken, every confirmed identity pair is passed
through src.sar_ais_quality.screen_identity_pairs -- the bracket check and the
alternative-vessel check that were validated 5/5 against the India tail
outliers (src/inspect_india_tail_outliers.py). Pairs it rejects are logged as
`suspected_misattribution` and excluded from the distribution the cutoffs are
derived from. This is NOT an India special case: GFW's SAR<->AIS correlator
mis-attributes MMSIs by tens of km wherever coastal traffic is dense with
patchy AIS, and one 97 km bad pair moved India's likely_spoofed from ~20 km to
~79 km. The same screen belongs in real detection runs later, so it lives in
its own module.

What it does NOT do
------------------
It does not re-fit or replace match.py's distance maths or the ST-DBSCAN
implementation -- it calls the existing code and only derives the numbers those
functions take as parameters. It is a read-only calibration layer whose single
side effect is writing data/processed/threshold_calibration_{region}.json.

Run:
  python -m src.calibrate_thresholds                  # every region whose raw
                                                      # data is already cached
                                                      # (-> the Gulf validation)
  python -m src.calibrate_thresholds hormuz_control   # one region
  python -m src.calibrate_thresholds india_mangalore --fetch   # allow live GFW pull
"""
import json
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

PROC = "data/processed"
RAW = "data/raw"

# ======================================================================
# SAR-vs-AIS distance percentile choices -- justification
# (house style, cf. trajectory_predict.py / spatiotemporal_cluster.py:
#  state the reasoning, not just the number)
# ======================================================================
#
# WHICH POPULATION. The raw nearest-AIS distance over ALL SAR detections in a
# region is a MIXTURE: (a) real vessels that are correctly broadcasting AIS
# (true pairs, distance = pure measurement floor) and (b) real vessels that
# are genuinely AIS-dark that hour, whose "nearest AIS" is some unrelated ship
# elsewhere in the bin (distance = tens of km, and NOT a spoofing signal). A
# percentile of that mixture thresholds neither component -- on the Gulf
# control set the 90th percentile of the full population is ~23 km, dragged
# there by the dark-vessel tail, nowhere near the hand value.
#
# GeoTrackNet calibrates its detection contour on the NORMAL population only
# (reconstructed AIS tracks), not on a mixture. The equivalent normal
# population here is the set of CONFIRMED same-vessel pairs: SAR detections
# that carry an MMSI and were matched by interpolating that same vessel's own
# AIS track to the SAR timestamp (match.py's identity path; `time_gap_seconds`
# is non-null exactly for these). Their distance spread is the region's real
# measurement floor -- grid quantisation (~1.1 km cells, config.py), up to a
# half-hour SAR-vs-bin time offset bridged by interpolation over coarse hourly
# samples, and SAR geolocation error -- with no dark-vessel contamination.
# BOTH cutoffs below are percentiles of THAT distribution.
#
# MATCHED_PERCENTILE = 50.0  (median of the confirmed-pair distribution)
#   The median offset of a known-true pair is this region's central
#   measurement floor. A pair tighter than the typical true pair is at least
#   as consistent with identity as a coin-flip of real matches -> "matched".
#   Using the median (not a low percentile) keeps this a non-cherry-picked
#   statistic; on the Gulf control it lands at 2.8 km vs the 1.5 km hand value
#   (1.9x -- the hand value was a tight "one grid cell" pick; the data says
#   the real floor in an hourly-AIS region is a touch wider).
#
# LIKELY_SPOOFED_FALSE_ALARM_RATE = 0.01  -> the (1 - FAR) = 99th percentile
#   The a-contrario cutoff, GeoTrackNet-style: a CONFIRMED same-vessel pair
#   exceeds this separation only ~1 time in 100. Beyond it, "same vessel, just
#   measurement noise" is exhausted as an explanation -> "likely_spoofed".
#   Exposed AS a false-alarm rate so an operator can trade recall for
#   precision without touching code. Why 1% and not GeoTrackNet's usual 0.1%:
#   the confirmed-pair baseline is small (tens-to-low-hundreds of pairs on a
#   single control window), so the 99.9th percentile would just be the sample
#   maximum and unstable. Drop FAR to 0.001 once a multi-week baseline with
#   >~1000 confirmed pairs is available. On the Gulf control, 1% FAR lands at
#   21.7 km vs the 20 km hand value (1.08x).
#
# The "discrepant" tier is not separately calibrated: it is simply the band
# matched_km < d <= likely_spoofed_km -- the same inconclusive-by-design
# middle as in match.py.
#
# MIN_IDENTITY_PAIRS: below this many confirmed pairs the derived distance
# cutoffs are reported but flagged low-confidence (single short control
# window rather than a proper multi-week baseline).
MATCHED_PERCENTILE = 50.0
LIKELY_SPOOFED_FALSE_ALARM_RATE = 0.01
LIKELY_SPOOFED_PERCENTILE = 100.0 * (1.0 - LIKELY_SPOOFED_FALSE_ALARM_RATE)
MIN_IDENTITY_PAIRS = 200

# ======================================================================
# ST-DBSCAN epsilon derivation -- adaptive, per Bai et al.
# ======================================================================
#
# eps_spatial: Bai et al.'s adaptive DBSCAN sets eps from the KNEE of the
#   sorted k-nearest-neighbour distance curve (k = min_pts - 1). ST-DBSCAN
#   here runs at min_pts = 2, so k = 1: for every AIS position we take the
#   haversine distance to the nearest OTHER vessel present in the SAME hourly
#   bin (a "contemporaneous nearest-neighbour distance"). That distribution
#   encodes the region's traffic geometry -- lane spacing, anchorage packing,
#   density. Its knee is the separation at which points stop being "in a
#   group" and become "just the ambient spacing": the natural eps for THIS
#   region. The knee is located with the standard max-distance-to-chord
#   method on the sorted curve (Kneedle-lite). Alternative estimators (p75,
#   p90, mean+1std) are reported alongside so the choice is auditable and can
#   be cross-checked against the Gulf hand value.
#
#   Floor: the AIS presence grid is ~1.1 km and trajectory_predict.py's
#   "notable" tier already admits multi-km route flex with zero spoofing, so
#   eps_spatial is clamped to >= EPS_SPATIAL_FLOOR_KM -- below that it would
#   just chase cell quantisation, the same lower bound spatiotemporal_cluster
#   argues by hand.
#
# eps_temporal: the temporal analogue is the per-vessel gap between successive
#   AIS bins. AIS is hourly, so this is ~1 h by construction; its high
#   percentile is the longest "still one continuous presence" gap the
#   region's coverage produces. eps_temporal = ceil(EPS_TEMPORAL_PERCENTILE
#   (90th) of that gap distribution), clamped to >= 2x the temporal
#   resolution (i.e. >= 2 h) so a single dropped bin never splits one
#   episode. This reproduces the hand-set 2 h wherever coverage is hourly and
#   only widens where the region's AIS is genuinely sparser.
#
# min_pts stays 2 (single-linkage first stage). It is a structural choice in
# spatiotemporal_cluster.py, not a region-scaled quantity, so it is passed
# through unchanged and only recorded in the report.
EPS_SPATIAL_FLOOR_KM = 3.0
EPS_SPATIAL_NN_PERCENTILES = [10, 25, 50, 75, 90]
EPS_TEMPORAL_PERCENTILE = 90.0
DEFAULT_MIN_PTS = 2
DEFAULT_TEMPORAL_RESOLUTION_HOURS = 1.0

# "reasonably close" for the automatic-vs-manual agreement check: within a
# factor of 2 either way is AGREES, within a factor of 3 is LOOSE, beyond
# that is DIVERGES. A calibration that AGREES on every parameter reproduces
# the hand work; anything else is surfaced for review rather than trusted
# silently.
AGREE_RATIO = 2.0
LOOSE_RATIO = 3.0

# ======================================================================
# Region registry
# ======================================================================
REGIONS = {
    # Gulf validation region. bbox and window match the already-pulled Hormuz
    # control set (the "mar2026" filename actually holds the Jan 17-21 2026
    # control window -- see reclassify_hormuz.py). manual_reference carries the
    # hand-calibrated numbers we already validated by eye, so this region
    # doubles as the automatic-vs-manual agreement check.
    "hormuz_control": {
        "bbox": {"min_lat": 25.5, "max_lat": 27.0, "min_lon": 55.5, "max_lon": 57.5},
        "start_date": "2026-01-17",
        "end_date": "2026-01-21",
        "raw_label": "hormuz_control_mar2026",
        "manual_reference": {
            "matched_km": 1.5,
            "likely_spoofed_km": 20.0,
            "eps_spatial_km": 8.0,
            "eps_temporal_hours": 2.0,
        },
    },
    # First new region: New Mangalore approaches + outer anchorages.
    #
    # Baseline window: 2025-09-01 .. 2025-10-12 (6 weeks). Widened from the
    # original 14-day window because the Gulf validation flagged LOW
    # confidence at 77 confirmed pairs over 5 days, and India has NO manual
    # reference to fall back on -- the baseline has to be solid on its own.
    # Incident-free check (done 2026-09-07):
    #   - SELENIA deep-history review: its New Mangalore shuttle ran early
    #     Sept -> early Nov 2025 with every India call classified routine, no
    #     flagged anomaly.
    #   - Swept all ~20 deep-history vessels for any event inside THIS bbox
    #     during THIS window: only SELENIA appears, and only as ordinary
    #     `port_visit` events (no loitering, no anomaly tier).
    #   - Every known incident in the project (Qatar GNSS spoofing Oct 3-7
    #     2025; Hormuz crisis Mar 2026; Hormuz control Jan 2026) is in the
    #     Persian Gulf / Strait of Hormuz -- ~1,500 km away, a different sea,
    #     bbox does not intersect. The Qatar window overlaps in TIME only,
    #     not in space.
    # No hand-calibrated thresholds exist for India -> manual_reference is
    # None and the report flags every derived value as provisional.
    "india_mangalore": {
        "bbox": {"min_lat": 12.0, "max_lat": 13.8, "min_lon": 73.8, "max_lon": 75.2},
        "start_date": "2025-09-01",
        "end_date": "2025-10-12",
        "raw_label": "india_mangalore_sep_oct2025",
        "manual_reference": None,
    },
    # Second new region: Jamnagar refinery waters + Vadinar terminal, Gulf of
    # Kutch. Major Russian-crude import destination (Nayara/Vadinar is EU- +
    # UK-sanctioned since the Jul 2025 18th package); chronic shadow-fleet /
    # deceptive-AIS background is expected -- "incident-free" here means "no
    # discrete event", not "no spoofing".
    # Incident-free check (2026-09-08): no documented discrete maritime
    # incident (attack / explosion / fire / spill / grounding / collision /
    # detention / port closure) inside this bbox/window in 2025-2026. Nearby
    # but NOT in-bbox/in-window: Operation Sindoor hostilities early-mid May
    # 2025 (land strikes ~150 km NE, no maritime effect); MV Rajeev Gandhi
    # container loss off Okha Jun 2026 (west of the bbox, en route to Mundra);
    # 2025 Kerala oil spill (~1000 km S). Ping Shun (IMO 9231901) broadcast
    # Vadinar as destination then diverted ~30 Mar 2026 -- signalling only.
    # Baseline window 2025-09-01..2025-10-12: 6 weeks, same width/timing as
    # india_mangalore (deliberate parallel), clear of Operation Sindoor, 3
    # SAR passes with ~92% AIS-match rate. No hand-calibrated thresholds
    # exist -> manual_reference is None, every derived value stays PROVISIONAL.
    "jamnagar_vadinar": {
        "bbox": {"min_lat": 22.2, "max_lat": 22.7, "min_lon": 69.5, "max_lon": 70.1},
        "start_date": "2025-09-01",
        "end_date": "2025-10-12",
        "raw_label": "jamnagar_vadinar_sep_oct2025",
        "manual_reference": None,
    },
}


# ======================================================================
# Data loading
# ======================================================================
def _load_raw(region_key, cfg, allow_fetch):
    """Return (sar_df, ais_df, source_description).

    Prefer already-pulled data/raw/{raw_label}_*.csv. Only hit the GFW API if
    the cache is absent AND allow_fetch is True, so a bare
    `python -m src.calibrate_thresholds` never makes a surprise network call.
    """
    label = cfg["raw_label"]
    sar_path = f"{RAW}/{label}_sar_detections.csv"
    ais_path = f"{RAW}/{label}_ais_positions.csv"

    if os.path.exists(sar_path) and os.path.exists(ais_path):
        sar_df = pd.read_csv(sar_path, parse_dates=["timestamp"])
        ais_df = pd.read_csv(ais_path, parse_dates=["timestamp"])
        return sar_df, ais_df, f"cached raw: {sar_path} + {ais_path}"

    if not allow_fetch:
        raise FileNotFoundError(
            f"no cached raw data for '{region_key}' ({sar_path}). "
            f"re-run with --fetch to pull it from GFW "
            f"(bbox {cfg['bbox']}, {cfg['start_date']}..{cfg['end_date']})"
        )

    # Live pull -- exactly the fetch_sar.py / fetch_ais.py path, cached for reuse.
    from src.fetch_sar import fetch_sar_detections
    from src.fetch_ais import fetch_ais_positions

    sar_df = fetch_sar_detections(cfg["bbox"], cfg["start_date"], cfg["end_date"])
    ais_df = fetch_ais_positions(cfg["bbox"], cfg["start_date"], cfg["end_date"])
    sar_df.to_csv(sar_path, index=False)
    ais_df.to_csv(ais_path, index=False)
    return sar_df, ais_df, f"live GFW pull -> cached at {sar_path} + {ais_path}"


# ======================================================================
# Step 1-2: raw SAR-vs-AIS distance distribution -> distance thresholds
# ======================================================================
_PCT_GRID = {"p10": 10, "p25": 25, "p50": 50, "p75": 75,
             "p90": 90, "p95": 95, "p99": 99, "p99.9": 99.9}


def _pct_block(d):
    b = {k: round(float(np.percentile(d, p)), 4) for k, p in _PCT_GRID.items()}
    b["max"] = round(float(d.max()), 4)
    b["mean"] = round(float(d.mean()), 4)
    return b


def calibrate_distance_thresholds(sar_df, ais_df):
    """Compute the raw nearest-AIS distance for every SAR detection using
    match.py's own distance maths, then derive matched / likely_spoofed
    cutoffs as percentiles of the CONFIRMED same-vessel distance distribution.
    No fixed cutoff (1.5 / 20 km) is applied anywhere here.
    """
    from src.match import match_and_classify

    # match_and_classify returns the per-detection nearest-AIS distance_km
    # (unbounded search, no distance cap). `time_gap_seconds` is non-null
    # exactly for the identity path: SAR detection had an MMSI and we
    # interpolated that same vessel's own AIS track to the SAR time. Those
    # rows are the confirmed same-vessel pairs -- the GeoTrackNet "normal"
    # population. We ignore the returned `classification` entirely.
    matched = match_and_classify(sar_df, ais_df)
    dist = pd.to_numeric(matched["distance_km"], errors="coerce")

    d_all = dist[dist >= 0].dropna().to_numpy(dtype=float)
    is_identity = matched["time_gap_seconds"].notna() & (dist >= 0) & dist.notna()

    # --- Mis-attribution screen (runs on EVERY calibration) -----------
    # GFW ships some SAR detections pre-matched to an MMSI and match.py's
    # identity path trusts that. In dense coastal traffic that attribution is
    # often wrong by tens of km -- verified 5/5 on the India tail
    # (src/inspect_india_tail_outliers.py). One 97 km bad pair moved India's
    # likely_spoofed from ~22 km to ~79 km. This is a general GFW-correlator
    # weakness, not India-specific, so every confirmed identity pair is
    # screened with the bracket + alternative-vessel checks
    # (src.sar_ais_quality) BEFORE any percentile is taken. Pairs the screen
    # rejects are logged as suspected_misattribution and excluded from the
    # distribution the cutoffs are derived from.
    from src.sar_ais_quality import screen_identity_pairs

    screen = screen_identity_pairs(matched, ais_df, is_identity)
    kept_ids = set(screen.loc[screen["keep"], "sar_id"])
    id_rows = matched[is_identity]
    d_id_pre = pd.to_numeric(id_rows["distance_km"], errors="coerce").to_numpy(dtype=float)
    d_id = pd.to_numeric(
        id_rows.loc[id_rows["sar_id"].isin(kept_ids), "distance_km"], errors="coerce"
    ).to_numpy(dtype=float)

    n_all = int(d_all.size)
    n_id_pre = int(d_id_pre.size)
    n_id = int(d_id.size)
    excl = screen.loc[~screen["keep"]].copy()

    out = {
        "method": (
            "GeoTrackNet-style percentile of the CONFIRMED same-vessel "
            "(identity-matched) nearest-AIS distance distribution; the full "
            "SAR population is a true-match + dark-vessel mixture and is "
            "reported only for context"
        ),
        "percentile_choice": {
            "matched_percentile": MATCHED_PERCENTILE,
            "matched_basis": "median offset of a confirmed same-vessel pair",
            "likely_spoofed_percentile": round(LIKELY_SPOOFED_PERCENTILE, 4),
            "likely_spoofed_false_alarm_rate": LIKELY_SPOOFED_FALSE_ALARM_RATE,
            "likely_spoofed_basis": "a-contrario: a confirmed pair exceeds this only FAR of the time",
        },
        "n_sar_detections": int(len(matched)),
        "n_pairs_with_distance": n_all,
        "n_confirmed_identity_pairs_pre_screen": n_id_pre,
        "n_confirmed_identity_pairs": n_id,
    }

    # Mis-attribution screen results (always present).
    out["misattribution_screen"] = {
        "method": (
            "src.sar_ais_quality.screen_identity_pairs -- bracket check "
            "(own AIS brackets the SAR pass yet stays > 15 km away, or no own "
            "AIS within +/-2 h) + alternative-vessel check (a different vessel "
            "<= 5 km from the SAR point and <= half the attributed distance)"
        ),
        "runs_on": "every calibration (GFW SAR->MMSI mis-attribution is a "
                   "general dense-traffic data-quality issue, not region-specific)",
        "n_excluded_suspected_misattribution": int(len(excl)),
        "excluded_pairs": [
            {
                "sar_id": r["sar_id"],
                "matched_mmsi": r["matched_mmsi"],
                "ship_name": r["ship_name"],
                "sar_timestamp": str(r["sar_timestamp"]),
                "distance_km": r["distance_km"],
                "time_gap_h": r["time_gap_h"],
                "bracket_status": r["bracket_status"],
                "nearest_own_bin_km": r["nearest_own_bin_km"],
                "better_alt_exists": bool(r["better_alt_exists"]),
                "best_alt_mmsi": r["best_alt_mmsi"],
                "best_alt_km": r["best_alt_km"],
                "reason": r["exclusion_reason"],
            }
            for _, r in excl.iterrows()
        ],
    }
    if n_id_pre:
        frac_excl = len(excl) / n_id_pre
        out["misattribution_screen"]["fraction_excluded"] = round(frac_excl, 3)
        if frac_excl > 0.20:
            out["misattribution_screen"]["warning"] = (
                f"screen removed {frac_excl:.0%} of identity pairs -- unusually "
                f"high; check GFW attribution quality / the AIS pull before "
                f"trusting the derived cutoffs"
            )

    if n_all:
        out["all_pairs_percentiles_km"] = _pct_block(d_all)
        out["all_pairs_percentiles_km"]["note"] = (
            "MIXTURE (true matches + genuine dark vessels) -- not used for "
            "thresholds"
        )

    if n_id == 0:
        out["error"] = (
            "no confirmed identity-matched pairs survived the mis-attribution "
            "screen -- cannot calibrate distance thresholds"
            if n_id_pre else
            "no confirmed identity-matched pairs in this region/period -- "
            "cannot calibrate distance thresholds (need SAR detections that "
            "carry an MMSI and have an overlapping own-vessel AIS track)"
        )
        return out

    if n_id_pre and n_id_pre != n_id:
        out["pre_screen_confirmed_pair_percentiles_km"] = _pct_block(d_id_pre)
        out["pre_screen_derived_thresholds_km"] = {
            "matched_max": round(float(np.percentile(d_id_pre, MATCHED_PERCENTILE)), 4),
            "likely_spoofed_min": round(
                float(np.percentile(d_id_pre, LIKELY_SPOOFED_PERCENTILE)), 4),
        }

    out["confirmed_pair_percentiles_km"] = _pct_block(d_id)
    matched_km = float(np.percentile(d_id, MATCHED_PERCENTILE))
    spoofed_km = float(np.percentile(d_id, LIKELY_SPOOFED_PERCENTILE))
    out["derived_thresholds_km"] = {
        "matched_max": round(matched_km, 4),
        "likely_spoofed_min": round(spoofed_km, 4),
        "discrepant_band": [round(matched_km, 4), round(spoofed_km, 4)],
    }

    # Tail-robustness check on the likely_spoofed cutoff. The confirmed-pair
    # population can carry a few gross outliers -- a genuine position
    # displacement that happened during the "clean" baseline, or a GFW
    # SAR->MMSI attribution error (its correlator is not perfect in dense
    # coastal traffic). Those sit exactly where the 99th percentile is read,
    # so with only ~100-200 pairs a handful of them drags likely_spoofed_min
    # up by a large factor. `matched_max` (the median) is immune; this block
    # exists so the report can say when the spoof cutoff is tail-driven.
    #   - Tukey "far out" fence: Q3 + 3*IQR. Pairs beyond it are the outliers.
    #   - p95 is reported as a less tail-sensitive alternative reference.
    q1, q3 = np.percentile(d_id, [25, 75])
    iqr = q3 - q1
    far_fence = float(q3 + 3.0 * iqr)
    n_far_out = int((d_id > far_fence).sum())
    p95 = float(np.percentile(d_id, 95))
    heavy_tail = spoofed_km > 2.0 * p95 and n_far_out > 0
    out["likely_spoofed_tail_check"] = {
        "p95_km": round(p95, 4),
        "p99_km": round(spoofed_km, 4),
        "p99_over_p95": round(spoofed_km / p95, 2) if p95 > 0 else None,
        "tukey_far_out_fence_km": round(far_fence, 4),
        "n_pairs_beyond_fence": n_far_out,
        "heavy_tail": bool(heavy_tail),
        "note": (
            (f"HEAVY TAIL: likely_spoofed_min ({spoofed_km:.1f} km) is set by "
             f"{n_far_out} outlier pair(s) far beyond the bulk (p95 = "
             f"{p95:.1f} km). Those pairs are either genuine displacement in "
             f"the baseline or GFW SAR->MMSI mis-attribution -- the manual "
             f"spot-check must inspect each one and decide, since that "
             f"decision moves the cutoff between ~p95 and ~p99.")
            if heavy_tail else
            "no gross outliers beyond the Tukey far-out fence; p99 is not "
            "tail-driven"
        ),
    }

    out["confidence"] = "ok" if n_id >= MIN_IDENTITY_PAIRS else "low"
    if n_id < MIN_IDENTITY_PAIRS:
        out["confidence_note"] = (
            f"only {n_id} confirmed identity pairs (< {MIN_IDENTITY_PAIRS}); "
            f"this is a single short control window, not a multi-week "
            f"baseline. The {LIKELY_SPOOFED_PERCENTILE:g}th percentile is near "
            f"the sample maximum -- treat likely_spoofed_min as indicative. "
            f"Pull a longer baseline and drop the false-alarm rate to 0.001 "
            f"for an operational value."
        )
    return out


# ======================================================================
# Step 3: ST-DBSCAN epsilon derivation from the region's trajectory data
# ======================================================================
def _contemporaneous_nn_distances(ais_df):
    """For each AIS position, haversine km to the nearest OTHER vessel present
    in the same hourly bin. Returns a 1-D array (one entry per position that
    had at least one contemporaneous neighbour of a different MMSI)."""
    from src.match import haversine_km

    df = ais_df.dropna(subset=["lat", "lon", "timestamp", "mmsi"]).copy()
    df["hour_bin"] = df["timestamp"].dt.floor("h")

    nn = []
    for _, g in df.groupby("hour_bin", sort=False):
        if len(g) < 2:
            continue
        lat = g["lat"].to_numpy(dtype=float)
        lon = g["lon"].to_numpy(dtype=float)
        mmsi = g["mmsi"].to_numpy()
        for i in range(len(g)):
            dk = haversine_km(lat[i], lon[i], lat, lon)
            dk[mmsi == mmsi[i]] = np.inf  # mask self and same-vessel points
            m = dk.min()
            if np.isfinite(m):
                nn.append(float(m))
    return np.asarray(nn, dtype=float)


def _per_vessel_bin_gaps_hours(ais_df):
    """Hours between successive AIS bins for each vessel, pooled. Unit-safe
    against pandas 3 datetime64[us] (dividing by np.timedelta64(1,'h') gives
    float hours regardless of the underlying resolution -- a raw .astype
    would be 1000x off)."""
    df = ais_df.dropna(subset=["timestamp", "mmsi"]).sort_values(["mmsi", "timestamp"])
    gaps = []
    for _, g in df.groupby("mmsi", sort=False):
        if len(g) < 2:
            continue
        t = g["timestamp"].to_numpy()
        dh = np.diff(t) / np.timedelta64(1, "h")
        gaps.append(dh[dh > 0])
    return np.concatenate(gaps) if gaps else np.asarray([], dtype=float)


def _knee_of_curve(vals):
    """Kneedle-lite: value at the point on the sorted (ascending) curve
    farthest from the chord joining its two endpoints."""
    y = np.sort(np.asarray(vals, dtype=float))
    n = y.size
    if n < 3:
        return float(y[-1]) if n else float("nan")
    x = np.arange(n, dtype=float)
    x0, x1, y0, y1 = x[0], x[-1], y[0], y[-1]
    denom = math.hypot(x1 - x0, y1 - y0)
    if denom == 0:
        return float(y[-1])
    dist = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / denom
    return float(y[int(np.argmax(dist))])


def calibrate_st_dbscan_params(ais_df,
                               temporal_resolution_hours=DEFAULT_TEMPORAL_RESOLUTION_HOURS):
    """Derive eps_spatial / eps_temporal from the region's own AIS trajectory
    distribution (Bai et al. adaptive DBSCAN). min_pts is structural and
    passed through unchanged."""
    nn = _contemporaneous_nn_distances(ais_df)
    gaps = _per_vessel_bin_gaps_hours(ais_df)

    out = {
        "method_spatial": (
            "Bai et al. adaptive DBSCAN: knee (Kneedle-lite) of the sorted "
            "1-nearest contemporaneous inter-vessel distance curve, floored "
            f"at {EPS_SPATIAL_FLOOR_KM} km"
        ),
        "method_temporal": (
            f"ceil of the {EPS_TEMPORAL_PERCENTILE:g}th percentile of the "
            "per-vessel successive-AIS-bin gap, floored at 2x temporal "
            "resolution"
        ),
        "min_pts": DEFAULT_MIN_PTS,
        "min_pts_note": "structural (single-linkage first stage), not region-scaled",
        "n_ais_positions": int(len(ais_df)),
        "n_vessels": int(pd.to_numeric(ais_df["mmsi"], errors="coerce").dropna().nunique()),
        "n_contemporaneous_pairs": int(nn.size),
        "n_bin_gaps": int(gaps.size),
    }

    if nn.size >= 3:
        knee = _knee_of_curve(nn)
        eps_spatial = max(knee, EPS_SPATIAL_FLOOR_KM)
        out["nn_distance_percentiles_km"] = {
            f"p{p}": round(float(np.percentile(nn, p)), 4)
            for p in EPS_SPATIAL_NN_PERCENTILES
        }
        out["eps_spatial_candidates_km"] = {
            "knee": round(knee, 4),
            "p75": round(float(np.percentile(nn, 75)), 4),
            "p90": round(float(np.percentile(nn, 90)), 4),
            "mean_plus_1std": round(float(nn.mean() + nn.std()), 4),
        }
        out["eps_spatial_km"] = {
            "chosen_method": "knee",
            "knee_raw": round(knee, 4),
            "floor_km": EPS_SPATIAL_FLOOR_KM,
            "value": round(eps_spatial, 4),
        }
    else:
        out["eps_spatial_km"] = {
            "error": "too few contemporaneous inter-vessel pairs to derive a knee"
        }

    if gaps.size >= 3:
        gap_p = float(np.percentile(gaps, EPS_TEMPORAL_PERCENTILE))
        eps_temporal = max(math.ceil(gap_p), 2.0 * temporal_resolution_hours)
        out["bin_gap_hours"] = {
            "p50": round(float(np.percentile(gaps, 50)), 3),
            f"p{EPS_TEMPORAL_PERCENTILE:g}": round(gap_p, 3),
            "p99": round(float(np.percentile(gaps, 99)), 3),
        }
        out["eps_temporal_hours"] = {
            "percentile_raw": round(gap_p, 3),
            "floor_hours": 2.0 * temporal_resolution_hours,
            "value": round(float(eps_temporal), 3),
        }
    else:
        out["eps_temporal_hours"] = {"error": "too few successive-bin gaps"}

    return out


# ======================================================================
# Step 4-5: manual comparison + report assembly
# ======================================================================
def _agreement(derived, manual):
    if derived is None or manual is None or manual == 0:
        return None
    ratio = derived / manual
    if 1 / AGREE_RATIO <= ratio <= AGREE_RATIO:
        verdict = "AGREES"
    elif 1 / LOOSE_RATIO <= ratio <= LOOSE_RATIO:
        verdict = "LOOSE"
    else:
        verdict = "DIVERGES"
    return {
        "derived": round(derived, 4),
        "manual": manual,
        "ratio_derived_over_manual": round(ratio, 3),
        "abs_diff": round(derived - manual, 4),
        "verdict": verdict,
    }


def calibrate_region(region_key, allow_fetch=False):
    if region_key not in REGIONS:
        raise KeyError(f"unknown region '{region_key}'. known: {sorted(REGIONS)}")
    cfg = REGIONS[region_key]

    sar_df, ais_df, source = _load_raw(region_key, cfg, allow_fetch)
    dist_cal = calibrate_distance_thresholds(sar_df, ais_df)
    dbscan_cal = calibrate_st_dbscan_params(ais_df)

    derived = {
        "matched_km": dist_cal.get("derived_thresholds_km", {}).get("matched_max"),
        "likely_spoofed_km": dist_cal.get("derived_thresholds_km", {}).get("likely_spoofed_min"),
        "eps_spatial_km": dbscan_cal.get("eps_spatial_km", {}).get("value"),
        "eps_temporal_hours": dbscan_cal.get("eps_temporal_hours", {}).get("value"),
    }

    manual = cfg.get("manual_reference")
    if manual:
        per_param = {
            k: _agreement(derived.get(k), manual.get(k))
            for k in ("matched_km", "likely_spoofed_km",
                      "eps_spatial_km", "eps_temporal_hours")
        }
        verdicts = [v["verdict"] for v in per_param.values() if v]
        if verdicts and all(v == "AGREES" for v in verdicts):
            overall = "AGREES"
        elif verdicts and all(v in ("AGREES", "LOOSE") for v in verdicts):
            overall = "MOSTLY-AGREES"
        else:
            overall = "REVIEW"
        comparison = {
            "available": True,
            "reference_source": (
                "hand-calibrated Gulf values validated by eye "
                "(match.py 1.5/20 km, spatiotemporal_cluster.py 8 km / 2 h)"
            ),
            "agree_ratio": AGREE_RATIO,
            "per_parameter": per_param,
            "overall": overall,
        }
    else:
        scr = dist_cal.get("misattribution_screen", {})
        n_excl = scr.get("n_excluded_suspected_misattribution", 0)
        n_id = dist_cal.get("n_confirmed_identity_pairs", 0)
        _tc = dist_cal.get("likely_spoofed_tail_check", {})
        heavy = _tc.get("heavy_tail", False)
        comparison = {
            "available": False,
            "status": (
                "PROVISIONAL -- tail-outlier concern RESOLVED (automated); "
                "eps/geometry spot-check still outstanding"
            ),
            "note": (
                "No hand-calibrated thresholds exist for this region, so "
                "manual_reference stays unset and these values are not yet "
                "cleared for operational classification. "
                "RESOLVED: the SAR<->AIS mis-attribution / tail-outlier "
                "concern. calibrate_thresholds now auto-screens every identity "
                f"pair (src.sar_ais_quality) and dropped {n_excl} "
                "suspected-misattribution pair(s) this run BEFORE deriving the "
                "cutoffs; the likely_spoofed tail is "
                + ("still flagged heavy -- investigate before proceeding"
                   if heavy else
                   "no longer heavy (see likely_spoofed_tail_check).")
            ),
            "resolved": [
                "tail-outlier / GFW SAR->AIS mis-attribution: now screened "
                "automatically on every calibration run (bracket + "
                "alternative-vessel checks). Method validated 5/5 against the "
                "original India tail in src/inspect_india_tail_outliers.py."
            ],
            "remaining_checks_before_promotion": [
                f"Sample ~8-10 KEPT confirmed pairs ({n_id} available) across "
                f"the tiers on a folium map (SAR point + own +/-2 h track), "
                "same visual sanity check used for qatar_gnss_spoofing_oct2025 "
                "/ hormuz_control; confirm near-cutoff 'matched' pairs are "
                "visibly the same vessel.",
                "Sanity-check eps_spatial against the region's lane / "
                "anchorage geometry -- it must not bridge the inbound vs "
                "outbound approach lanes.",
                "Sanity-check eps_temporal: New Mangalore's is 6 h (vs the "
                "Gulf's 2 h) from sparser AIS coverage -- confirm that "
                "reflects real coverage and note the effect on ST-DBSCAN "
                "temporal permissiveness.",
                "Only after those, fill in manual_reference and drop the "
                "PROVISIONAL status.",
            ],
        }

    return {
        "region": region_key,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "bbox": cfg["bbox"],
        "baseline_period": [cfg["start_date"], cfg["end_date"]],
        "data_source": source,
        "sar_ais_distance_calibration": dist_cal,
        "st_dbscan_calibration": dbscan_cal,
        "derived_thresholds": derived,
        "manual_comparison": comparison,
    }


# ======================================================================
# Output
# ======================================================================
def _fmt(v, nd=3):
    return "n/a" if v is None else f"{v:.{nd}f}"


def write_report(report):
    path = f"{PROC}/threshold_calibration_{report['region']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return path


def print_report(report, path):
    d = report["derived_thresholds"]
    dist = report["sar_ais_distance_calibration"]
    db = report["st_dbscan_calibration"]

    print(f"\n{'=' * 72}")
    print(f"THRESHOLD CALIBRATION -- {report['region']}")
    print(f"{'=' * 72}")
    print(f"  bbox             {report['bbox']}")
    print(f"  baseline period  {report['baseline_period'][0]} .. {report['baseline_period'][1]}")
    print(f"  data source      {report['data_source']}")
    n_pre = dist.get("n_confirmed_identity_pairs_pre_screen",
                     dist.get("n_confirmed_identity_pairs", 0))
    print(f"  SAR detections   {dist.get('n_sar_detections', 0)}  "
          f"(with a distance: {dist.get('n_pairs_with_distance', 0)}, "
          f"confirmed same-vessel: {n_pre}, "
          f"confidence: {dist.get('confidence', 'n/a')})")
    print(f"  AIS positions    {db.get('n_ais_positions', 0)}  "
          f"({db.get('n_vessels', 0)} vessels, "
          f"{db.get('n_contemporaneous_pairs', 0)} contemporaneous pairs)")

    scr = dist.get("misattribution_screen")
    if scr:
        n_ex = scr.get("n_excluded_suspected_misattribution", 0)
        n_used = dist.get("n_confirmed_identity_pairs", 0)
        print(f"  mis-attrib screen  {n_pre} confirmed pairs -> excluded {n_ex} "
              f"suspected mis-attribution -> {n_used} used for thresholds")
        for e in scr.get("excluded_pairs", []):
            print(f"      - sar_id {e['sar_id']}  {e.get('ship_name') or '?'} "
                  f"({e['matched_mmsi']})  {e['distance_km']} km  "
                  f"gap {e['time_gap_h']} h  [{e['reason']}]")
        if scr.get("warning"):
            print(f"      ! {scr['warning']}")

    allp = dist.get("all_pairs_percentiles_km")
    if allp:
        print(f"  full-pop dist    p50={allp['p50']} p90={allp['p90']} "
              f"max={allp['max']} km  (mixture -- NOT thresholded)")
    pre = dist.get("pre_screen_derived_thresholds_km")
    if pre:
        print(f"  BEFORE screen    matched {pre['matched_max']} km  /  "
              f"likely_spoofed {pre['likely_spoofed_min']} km")

    print("\n  DERIVED THRESHOLDS  (from the confirmed same-vessel distance distribution)")
    print(f"    matched          d <= {_fmt(d['matched_km'])} km"
          f"          [p{MATCHED_PERCENTILE:g} = median confirmed-pair offset]")
    print(f"    discrepant       {_fmt(d['matched_km'])} < d <= {_fmt(d['likely_spoofed_km'])} km"
          f"   [inconclusive middle band]")
    print(f"    likely_spoofed   d >  {_fmt(d['likely_spoofed_km'])} km"
          f"          [p{LIKELY_SPOOFED_PERCENTILE:g}  ==  "
          f"{LIKELY_SPOOFED_FALSE_ALARM_RATE:g} false-alarm rate]")
    print(f"    ST-DBSCAN eps    {_fmt(d['eps_spatial_km'])} km spatial / "
          f"{_fmt(d['eps_temporal_hours'])} h temporal")
    print(f"                     [knee of NN-distance curve / "
          f"{EPS_TEMPORAL_PERCENTILE:g}th pct bin-gap; min_pts={db.get('min_pts')}]")
    cands = db.get("eps_spatial_candidates_km")
    if cands:
        print(f"                     eps_spatial candidates: "
              + ", ".join(f"{k}={v}" for k, v in cands.items()))
    bg = db.get("bin_gap_hours")
    et = db.get("eps_temporal_hours", {})
    if bg and isinstance(et, dict) and et.get("value"):
        print(f"                     bin-gap p90={et.get('percentile_raw')} h "
              f"-> eps_temporal {et['value']} h "
              f"(2 h where AIS coverage is hourly; wider where it is sparser)")
    tc = dist.get("likely_spoofed_tail_check")
    if tc:
        print(f"    likely_spoofed tail: p95={tc['p95_km']} km  p99={tc['p99_km']} km  "
              f"(p99/p95 = {tc['p99_over_p95']}), "
              f"{tc['n_pairs_beyond_fence']} pair(s) past the far-out fence")
        if tc.get("heavy_tail"):
            print(f"    ! {tc['note']}")
    if dist.get("confidence_note"):
        print(f"    ! {dist['confidence_note']}")

    mc = report["manual_comparison"]
    print("\n  VALIDATION vs GULF MANUAL VALUES")
    if not mc["available"]:
        print(f"    {mc.get('status', 'NO MANUAL COMPARISON AVAILABLE FOR THIS REGION.')}")
        print(f"    {mc['note']}")
        for r in mc.get("resolved", []):
            print(f"    [resolved] {r}")
        checks = mc.get("remaining_checks_before_promotion") or mc.get("recommended_spot_check")
        if checks:
            print("    Remaining before promotion / manual_reference:")
            for i, step in enumerate(checks, 1):
                print(f"      {i}. {step}")
    else:
        print(f"    {'parameter':<22}{'derived':>10}{'manual':>9}{'ratio':>9}   verdict")
        print(f"    {'-' * 60}")
        labels = {
            "matched_km": "matched (km)",
            "likely_spoofed_km": "likely_spoofed (km)",
            "eps_spatial_km": "eps_spatial (km)",
            "eps_temporal_hours": "eps_temporal (h)",
        }
        for k, lab in labels.items():
            v = mc["per_parameter"].get(k)
            if not v:
                print(f"    {lab:<22}{'n/a':>10}")
                continue
            print(f"    {lab:<22}{v['derived']:>10}{v['manual']:>9}"
                  f"{v['ratio_derived_over_manual']:>9}   {v['verdict']}")
        print(f"    {'-' * 60}")
        print(f"    OVERALL: {mc['overall']}   "
              f"(AGREES = within {AGREE_RATIO:g}x of the hand value on every parameter)")

    print(f"\n  written: {path}")


# ======================================================================
# Main
# ======================================================================
def main(argv):
    allow_fetch = "--fetch" in argv
    positional = [a for a in argv if not a.startswith("-")]

    if positional:
        regions = positional
    else:
        # Bare command: calibrate every region whose raw data is already on
        # disk (so it runs the Gulf validation and never makes a network
        # call). Regions without a cache are listed as skipped.
        regions = []
        for k, cfg in REGIONS.items():
            if os.path.exists(f"{RAW}/{cfg['raw_label']}_sar_detections.csv"):
                regions.append(k)
            else:
                print(f"[skip] {k}: no cached raw data -- run "
                      f"`python -m src.calibrate_thresholds {k} --fetch` to pull it")

    for region_key in regions:
        try:
            report = calibrate_region(region_key, allow_fetch=allow_fetch)
        except (FileNotFoundError, KeyError) as e:
            print(f"[skip] {region_key}: {e}")
            continue
        path = write_report(report)
        print_report(report, path)


if __name__ == "__main__":
    main(sys.argv[1:])
