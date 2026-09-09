"""Per-vessel historical reliability, consolidated across every window run.

Every check in this project so far scores a single time window in isolation:
match.py / reclassify_hormuz.py classify one SAR pass, trajectory_predict.py
scores one window's tracks, spatiotemporal_cluster.py clusters one window.
Nothing remembers that a vessel flagged in Hormuz March 2026 was also the
worst offender in Qatar October 2025.

This module builds that memory. It reads the already-saved outputs of all
four windows, keys every flagged event to a vessel/MMSI, normalises the
flag count by how many windows the vessel was actually observable in, cross-
references GFW vessel-identity data for static risk factors, and produces
one explainable 0-100 score per vessel.

Read-only w.r.t. the pipeline: it consumes saved CSVs and the GFW identity
API. It writes data/processed/vessel_reliability_scores_v6.csv (+ an identity
cache) and leaves the v1..v5 files untouched for comparison. v6 is FINAL --
every vessel in the top-15 has now been type/age-verified; the dashboard
should read v6.

v6 closes the top-15 verification: HAKKAISAN (352003690, IMO 9376878) was the
last unverified top-15 vessel (GFW type "OTHER", no IMO). Confirmed by
VesselFinder / vesseltracker.com / MyShipTracking as a genuine Crude Oil
Tanker (VLCC, built 2009, LOA 333 m, Panama). Added to CONFIRMED_VESSEL_TYPES
(recorded, but NOT a small-utility match -> no behavioural down-weight) and to
vessel_age_cache.json; its age>15yr component becomes +20 (17 yr old). Only
change vs v5.

v5 adds one CONFIRMED_VESSEL_TYPES entry: OCEAN CENTURY (636025162, IMO
9435650) is a Tug -- VesselFinder / MarineTraffic / vesseltracker /
vesseltracking.net all agree -- but GFW's type field returned "NA", so v2's
auto-detection missed it and its behavioural score carried full weight all the
way through v4, where the new age component lifted it to #2 (85). v5 applies
the v2 small-utility-craft down-weight (behavioural x0.25) it should have had
since v2. Only change vs v4.

v4 fixes the vessel-age component. GFW's registry `builtYear` is unpopulated
for every vessel in this fleet (confirmed gap), so the age>15yr component had
been contributing 0 to every score. v4 adds a second build-year source:
data/processed/vessel_age_cache.json, a manually-curated cache (same one-off,
cache-once philosophy as gfw_identity_cache.json) of build years resolved by
IMO / MMSI from public vessel-registry pages. When GFW's builtYear is null the
cache value is used; `built_year_source` records which. NOTE: the cache is
populated by manual lookup only -- Equasis (the IMO-endorsed registry) forbids
API / automated extraction / bulk download / storage under its Conditions of
Registration, and MarineTraffic / VesselFinder block automated access, so a
fleet-wide sweep is not permissible. It currently covers the vessels needed
for the reliability figure plus LENORE.

v3 makes the FOC component two-tier: a flag in the narrow, specifically-cited
SHADOW_FLEET_FLAGS list keeps the full +20; a flag in the broad ITF_FOC_FLAGS
list only (Liberia, Panama, Marshall Islands, Malta, ... -- most legitimate
large commercial tonnage) drops to +5, since on its own it does not
discriminate real sanctions-evasion risk. New column: foc_list_matched.

v2 adds a vessel-type check: small utility craft (pilot vessels, tugs,
landing craft, patrol/service launches) have a normal operating pattern --
many short hops, hard stops at jetties, tight turns inside harbour limits --
that mimics the deviation signature this pipeline flags, with nothing to do
with spoofing. Such vessels are NOT excluded (a pilot boat could still be
spoofed) but are marked with a vessel_type_caveat and their behavioural
score component is down-weighted (see UTILITY_BEHAVIORAL_MULTIPLIER).

Run:  python -m src.vessel_history [--no-gfw] [--refresh-gfw] [--max-gfw N]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import GFW_API_BASE_URL, GFW_API_TOKEN

PROC = "data/processed"
OUTPUT_FILE = f"{PROC}/vessel_reliability_scores.csv"        # v1 -- left untouched
OUTPUT_FILE_V2 = f"{PROC}/vessel_reliability_scores_v2.csv"  # v2 -- left untouched
OUTPUT_FILE_V3 = f"{PROC}/vessel_reliability_scores_v3.csv"  # v3 -- left untouched
OUTPUT_FILE_V4 = f"{PROC}/vessel_reliability_scores_v4.csv"  # v4 -- left untouched
OUTPUT_FILE_V5 = f"{PROC}/vessel_reliability_scores_v5.csv"  # v5 -- left untouched
OUTPUT_FILE_V6 = f"{PROC}/vessel_reliability_scores_v6.csv"  # written by this module -- FINAL
GFW_CACHE_PATH = f"{PROC}/gfw_identity_cache.json"
AGE_CACHE_PATH = f"{PROC}/vessel_age_cache.json"

# --------------------------------------------------------------------------
# The four distinct time windows, in chronological order. `key` is the id
# used everywhere below; `order` gives recency ranking for
# most_recent_flag_window.
# --------------------------------------------------------------------------
WINDOWS = [
    {"key": "qatar_control_sep2025", "order": 1, "label": "Qatar control (Sep 2025)"},
    {"key": "qatar_spoofing_oct2025", "order": 2, "label": "Qatar GNSS spoofing (Oct 2025)"},
    {"key": "hormuz_control_jan2026", "order": 3, "label": "Hormuz control (Jan 2026)"},
    {"key": "hormuz_crisis_mar2026", "order": 4, "label": "Hormuz crisis (Mar 2026)"},
]
WINDOW_ORDER = {w["key"]: w["order"] for w in WINDOWS}
WINDOW_LABEL = {w["key"]: w["label"] for w in WINDOWS}
N_WINDOWS = len(WINDOWS)

# Confirmed false-positive patterns diagnosed earlier this project: a single
# isolated AIS ping far off-track for one hourly bin, then an immediate
# snap-back. Their "flagged" events are decoding artifacts, not behaviour, so
# they must not count toward any negative score. The vessels still appear in
# the table (with flagged_event_count forced to 0 and is_known_artifact=True)
# so the exclusion is visible rather than silent.
EXCLUDE_MMSIS = {257933000, 636015780}  # BELRAY, KEN STAR

# Vessels discussed earlier in the project -- always looked up and always
# given their own breakdown in the printout.
NAMED_VESSELS = {
    306531000: "LENORE",
    422034900: "OURA",
    422337200: "KAREN8",
    314001120: "AN LI",
}

# --------------------------------------------------------------------------
# Flag-of-convenience reference sets (ISO3 codes, matching GFW's `flag`).
# Kept as explicit, citable lists -- NOT a black box -- so a reviewer can see
# exactly why a vessel got the FOC penalty and adjust the set.
# --------------------------------------------------------------------------
# ITF Fair Practices Committee declared flags of convenience (the standard
# reference list).
ITF_FOC_FLAGS = {
    "ATG",  # Antigua & Barbuda
    "BHS",  # Bahamas
    "BRB",  # Barbados
    "BLZ",  # Belize
    "BMU",  # Bermuda (UK)
    "KHM",  # Cambodia
    "CYM",  # Cayman Islands (UK)
    "COM",  # Comoros
    "CYP",  # Cyprus
    "GNQ",  # Equatorial Guinea
    "FRA",  # French International Register (2nd register) -- ITF-listed
    "DEU",  # German International Ship Register -- ITF-listed
    "GEO",  # Georgia
    "GIB",  # Gibraltar (UK)
    "HND",  # Honduras
    "JAM",  # Jamaica
    "LBN",  # Lebanon
    "LBR",  # Liberia
    "MLT",  # Malta
    "MHL",  # Marshall Islands
    "MUS",  # Mauritius
    "MDA",  # Moldova
    "MNG",  # Mongolia
    "MMR",  # Myanmar
    "ANT",  # Netherlands Antilles (legacy code)
    "PAN",  # Panama
    "PRK",  # North Korea
    "STP",  # Sao Tome & Principe
    "VCT",  # St Vincent & the Grenadines
    "KNA",  # St Kitts & Nevis
    "LCA",  # St Lucia
    "LKA",  # Sri Lanka -- ITF-listed
    "TON",  # Tonga
    "VUT",  # Vanuatu
}
# Registries repeatedly named in 2023-2026 "dark fleet" / sanctions-evasion
# reporting (S&P Global, Lloyd's List, KSE Institute, UANI). Not classic ITF
# FOCs but the reflagging destinations for the shadow tanker fleet. This is
# the NARROW list -- a shadow_fleet match is a specific signal and keeps the
# full FOC weight (W_FOC_SHADOW); a match on the broad ITF list only
# (ITF_FOC_FLAGS minus this set) is real but weak and scores W_FOC_ITF_ONLY.
SHADOW_FLEET_FLAGS = {
    "COK",  # Cook Islands
    "GAB",  # Gabon
    "PLW",  # Palau
    "CMR",  # Cameroon
    "DJI",  # Djibouti
    "GUY",  # Guyana
    "SLE",  # Sierra Leone
    "TZA",  # Tanzania (Zanzibar)
    "TGO",  # Togo
    "BES",  # Bonaire/Sint Eustatius/Saba -- small registry used for recent
    #          tanker reflagging out of mainstream registers (e.g. LENORE)
    "SWZ",  # Eswatini (landlocked -- flag-of-convenience only)
}
FOC_FLAGS = ITF_FOC_FLAGS | SHADOW_FLEET_FLAGS

# --------------------------------------------------------------------------
# Reliability score -- deliberately additive and transparent (0-100).
# NOTE: higher score = LOWER reliability / MORE concern. Three components,
# each printed separately so the total is always decomposable:
#
#   behavioral_score  0-60  = 60 * min(1, flag_rate / FLAG_RATE_CAP)
#       flag_rate = flagged_event_count / total_windows_observed.
#       The cap is reached at FLAG_RATE_CAP flagged events per observed
#       window; behaviour is the primary signal so it carries the most
#       weight.
#   foc_score         0, 5 or 20 -- v3 two-tier (see component 2 in
#       score_vessels for the full reasoning):
#         +20 (W_FOC_SHADOW)   flag state in SHADOW_FLEET_FLAGS -- the narrow,
#             specifically-cited 2023-2026 dark-fleet reflagging list. A real,
#             specific signal; keeps full weight.
#         +5  (W_FOC_ITF_ONLY) flag state in the broad ITF_FOC_FLAGS list ONLY
#             (not also shadow-fleet) -- labour/tax classification that most
#             legitimate large commercial tonnage (Liberia, Panama, Marshall
#             Islands, Malta ...) also flies, so weak / non-discriminating on
#             its own. v2 gave this a flat +20, which fired on "large
#             commercial vessel" generally (13 of the v2 top-14 matched here).
#         0   otherwise.
#   age_score         0 or 20 = 20 if a build year is known AND vessel age
#       > AGE_THRESHOLD_YEARS, else 0 (unknown age scores 0 and is noted --
#       it never invents risk it cannot evidence).
#
#   reliability_score = behavioral_score + foc_score + age_score
# --------------------------------------------------------------------------
W_BEHAVIORAL = 60.0
W_FOC_SHADOW = 20.0    # flag in SHADOW_FLEET_FLAGS -- specific dark-fleet signal
W_FOC_ITF_ONLY = 5.0   # flag in the broad ITF_FOC_FLAGS list only -- weak signal
W_AGE = 20.0
FLAG_RATE_CAP = 4.0
AGE_THRESHOLD_YEARS = 15
CURRENT_YEAR = 2026  # project "today" is 2026-09-05

# Uniform "this event looks like spoofing" rule applied to every SAR-vs-AIS
# file. The Hormuz files already carry an explicit `likely_spoofed` label
# (added by reclassify_hormuz.py); the older Qatar files predate it, so the
# same 20 km separation threshold match.py now uses is reconstructed from
# `distance_km`. Net effect: one rule across all four windows.
SPOOF_DISTANCE_KM = 20.0

# An "event" must be a discrete incident, not a per-hour-bin tally. A vessel
# held in the `high` deviation tier for 6 consecutive hourly bins is ONE
# event, not six; several SAR blobs matched to one AIS track in one satellite
# pass are ONE spoofing event, not several. Consecutive occurrences within
# EVENT_GAP_HOURS are collapsed (same reasoning as the episode reduction in
# spatiotemporal_cluster.py).
EVENT_GAP_HOURS = 3.0

# --------------------------------------------------------------------------
# Small-utility-craft vessel-type check (v2)
# --------------------------------------------------------------------------
# Explicit category list -- printed at runtime, never applied silently. A
# resolved vessel type is normalised (lower-case, spaces/dashes/slashes ->
# "_") and matched against this set OR against the substring keywords in
# _is_small_utility(). Deliberately NARROW: it must catch pilot boats, tugs,
# landing craft, patrol/service launches -- NOT broad merchant buckets like
# GFW's "OTHER_NON_FISHING" (which also contains real cargo/tanker traffic).
SMALL_UTILITY_CRAFT_TYPES = {
    "pilot_vessel", "pilot", "pilot_boat",
    "tug", "tugboat", "towing_vessel", "pusher_tug",
    "landing_craft",
    "service_vessel", "service_craft", "utility_vessel", "utility_craft",
    "patrol_vessel", "patrol_boat", "patrol", "law_enforcement", "coast_guard",
    "port_tender", "tender", "crew_boat", "crew_tender",
    "workboat", "work_boat", "line_handling_boat", "mooring_vessel",
    "supply_vessel", "offshore_supply_vessel", "platform_supply_vessel", "osv",
    "dredger", "hopper_dredger", "pilot_patrol",
}
_UTILITY_KEYWORDS = ("pilot", "tug", "landing_craft", "patrol", "tender",
                     "workboat", "work_boat", "supply_vessel", "line_handling",
                     "mooring", "dredg", "crew_boat", "coast_guard")

# GFW's identity endpoint only returns coarse NN-inferred buckets ("OTHER",
# "CARGO", "TANKER") and the 4wings files only carry "OTHER_NON_FISHING" etc.
# -- neither distinguishes a pilot boat or landing craft from a merchant
# ship. These are finer types confirmed by manual external registry lookups
# this session and entered explicitly. They take precedence over any coarse
# API/CSV type. Extend this dict as more are confirmed.
CONFIRMED_VESSEL_TYPES = {
    422565000: "pilot_vessel",    # HADI 3       -- external registry lookup
    422405300: "landing_craft",   # AMIR BANDAR (IMO 9506370) -- external lookup
    636025162: "tug",             # OCEAN CENTURY (IMO 9435650) -- VesselFinder,
    #   MarineTraffic, vesseltracker.com and vesseltracking.net all list it as
    #   Tug / Towing Vessel (built 2007, GT 464, LOA ~37 m -- a small coastal
    #   tug). GFW returned type "NA", so v2's auto-detection missed it and its
    #   behavioural score kept full weight through v4. (2026-09-07 lookup.)
    352003690: "crude_oil_tanker",  # HAKKAISAN (IMO 9376878) -- VesselFinder,
    #   vesseltracker.com and MyShipTracking all list it as a Crude Oil Tanker
    #   / VLCC (built 2009, GT ~160,632, DWT ~309,708, LOA 333 m, Panama flag,
    #   callsign 3E5799). GFW returned type "OTHER" with no IMO, leaving it
    #   unverified in the v5 top-15. Confirmed here as a GENUINE large tanker:
    #   recorded so its type is explicit, but NOT a small-utility match, so no
    #   behavioural down-weight -- it just takes the age>15yr boost. (2026-09-07)
}
# Coarse type strings that are NOT specific enough to resolve a vessel type
# from (they get skipped, falling through to the next source).
_VAGUE_TYPES = {"", "other", "unknown", "insufficient_data", "other_non_fishing",
                "gear", "not_available", "none", "na", "n_a", "null"}

# --------------------------------------------------------------------------
# Small-utility-craft behavioural down-weight (justified like FOC/age above)
# --------------------------------------------------------------------------
# Pilot boats, tugs, landing craft and patrol/service launches operate, BY
# DESIGN, in the way this pipeline's deviation signature is tuned to catch:
# many short hops, hard stops alongside jetties, tight turns inside harbour
# limits, long idle periods, AIS gaps in sheds. For these craft a high
# flag_rate is the EXPECTED baseline of doing their job and carries almost no
# information about position manipulation -- but not *zero* (one could still
# be spoofed, or do something genuinely anomalous), so we down-weight rather
# than exclude.
#
# Only the BEHAVIOURAL component is multiplied by UTILITY_BEHAVIORAL_MULTIPLIER
# for caveated vessels. FOC and age are static identity facts, not behaviour,
# and are left unchanged. 0.25 mirrors the "~3 of 4 flags are class-explained"
# logic already baked into FLAG_RATE_CAP: a caveated craft needs roughly 4x
# the flag_rate of a merchant vessel to reach the same behavioural score.
UTILITY_BEHAVIORAL_MULTIPLIER = 0.25


def _norm_type(t):
    if not isinstance(t, str) or not t.strip():
        return None
    return (t.strip().lower()
            .replace(" ", "_").replace("-", "_").replace("/", "_"))


def _is_small_utility(t):
    n = _norm_type(t)
    if n is None:
        return False
    if n in SMALL_UTILITY_CRAFT_TYPES:
        return True
    return any(k in n for k in _UTILITY_KEYWORDS)


# ==========================================================================
# 1-2. Consolidate flagged events + window observations
# ==========================================================================
def _num_mmsi(series):
    return pd.to_numeric(series, errors="coerce")


def _count_episodes(timestamps):
    """Number of discrete events: runs of timestamps separated by more than
    EVENT_GAP_HOURS each count once."""
    t = pd.to_datetime(pd.Series(list(timestamps)), utc=True, errors="coerce").dropna()
    if t.empty:
        return 0
    t = t.sort_values()
    gaps_h = t.diff().dt.total_seconds().to_numpy()[1:] / 3600.0
    return 1 + int((gaps_h > EVENT_GAP_HOURS).sum())


def _name_map():
    names = {}
    for path, col in [
        (f"{PROC}/qatar_gnss_spoofing_oct2025_classified.csv", "ship_name"),
        (f"{PROC}/qatar_control_sept2025_classified.csv", "ship_name"),
        (f"{PROC}/hormuz_trajectory_deviation_v2.csv", "ship_name"),
        (f"{PROC}/hormuz_spatiotemporal_clusters.csv", "ship_name"),
    ]:
        df = pd.read_csv(path)
        key = "vessel_mmsi" if "vessel_mmsi" in df.columns else "mmsi"
        sub = df[[key, col]].copy()
        sub[key] = _num_mmsi(sub[key])
        sub = sub.dropna(subset=[key])
        for mmsi, nm in sub.groupby(key)[col].agg(
                lambda s: s.dropna().iloc[0] if s.notna().any() else None).items():
            names.setdefault(int(mmsi), nm)
    return names


def load_events():
    """Return a long DataFrame: one row per (vessel, window, source) with an
    observed flag and a flagged-event count."""
    rows = []

    def add(mmsi, window_key, source, observed=True, flagged_events=0, event_type=None):
        m = pd.to_numeric(pd.Series([mmsi]), errors="coerce").iloc[0]
        if pd.isna(m):
            return
        rows.append({
            "mmsi": int(m), "window": window_key, "source": source,
            "observed": observed, "flagged_events": int(flagged_events),
            "event_type": event_type,
        })

    # --- SAR-vs-AIS classified/reclassified files -------------------------
    sar_files = [
        (f"{PROC}/qatar_gnss_spoofing_oct2025_classified.csv", "qatar_spoofing_oct2025"),
        (f"{PROC}/qatar_control_sept2025_classified.csv", "qatar_control_sep2025"),
        (f"{PROC}/hormuz_crisis_mar2026_reclassified.csv", "hormuz_crisis_mar2026"),
        (f"{PROC}/hormuz_control_mar2026_reclassified.csv", "hormuz_control_jan2026"),
    ]
    for path, wkey in sar_files:
        df = pd.read_csv(path)
        # Vessel key: prefer own mmsi, else the AIS track the SAR blob matched.
        vessel = _num_mmsi(df["mmsi"])
        if "matched_mmsi" in df.columns:
            vessel = vessel.fillna(_num_mmsi(df["matched_mmsi"]))
        df = df.assign(_vessel=vessel).dropna(subset=["_vessel"])
        dist = pd.to_numeric(df.get("distance_km"), errors="coerce")
        df["_spoof"] = (df["classification"].astype(str) == "likely_spoofed") | \
                       (dist > SPOOF_DISTANCE_KM)
        ts_col = "sar_timestamp" if "sar_timestamp" in df.columns else None
        for mmsi, grp in df.groupby("_vessel"):
            spoof_rows = grp[grp["_spoof"]]
            # collapse multiple blobs in one satellite pass to one event
            n_events = (_count_episodes(spoof_rows[ts_col]) if ts_col
                        else len(spoof_rows))
            add(mmsi, wkey, Path(path).name, observed=True,
                flagged_events=n_events, event_type="sar_ais_likely_spoofed")

    # --- Trajectory-deviation (high tier only) --------------------------
    v2 = pd.read_csv(f"{PROC}/hormuz_trajectory_deviation_v2.csv")
    v2map = {"CONTROL (Jan 17-21, 2026)": "hormuz_control_jan2026",
             "CRISIS (Mar 19-24, 2026)": "hormuz_crisis_mar2026"}
    v2 = v2.assign(_wkey=v2["window"].map(v2map), _mmsi=_num_mmsi(v2["mmsi"]))
    v2 = v2.dropna(subset=["_wkey", "_mmsi"])
    for (mmsi, wkey), grp in v2.groupby(["_mmsi", "_wkey"]):
        high = grp[grp["tier"] == "high"]
        add(mmsi, wkey, "hormuz_trajectory_deviation_v2.csv", observed=True,
            flagged_events=_count_episodes(high["predicted_timestamp"]),
            event_type="trajectory_deviation_high")

    # --- Spatiotemporal clusters (genuine multi-vessel coherence) --------
    sc = pd.read_csv(f"{PROC}/hormuz_spatiotemporal_clusters.csv")
    scmap = {"CONTROL": "hormuz_control_jan2026", "CRISIS": "hormuz_crisis_mar2026"}
    sc = sc.assign(_wkey=sc["window"].map(scmap), _mmsi=_num_mmsi(sc["vessel_mmsi"]))
    sc = sc.dropna(subset=["_wkey", "_mmsi"])
    for (mmsi, wkey), grp in sc.groupby(["_mmsi", "_wkey"]):
        genuine = grp[grp["cluster_coherence_label"] == "genuine_multivessel_coherence"]
        add(mmsi, wkey, "hormuz_spatiotemporal_clusters.csv", observed=True,
            flagged_events=int(genuine["cluster_id"].nunique()),
            event_type="genuine_cluster_member")

    return pd.DataFrame(rows)


def consolidate(events):
    """Collapse the long event table to one row per vessel with the
    normalised history metrics from requirement 2."""
    per_vessel = []
    for mmsi, grp in events.groupby("mmsi"):
        is_artifact = mmsi in EXCLUDE_MMSIS
        windows_seen = sorted(grp.loc[grp["observed"], "window"].unique(),
                              key=lambda w: WINDOW_ORDER[w])
        # Artifact events are zeroed but the vessel is still recorded.
        flagged_by_window = (grp[~pd.Series(is_artifact, index=grp.index)]
                             .groupby("window")["flagged_events"].sum()
                             if not is_artifact else pd.Series(dtype=int))
        flagged_total = 0 if is_artifact else int(grp["flagged_events"].sum())
        n_windows = len(windows_seen)
        flag_rate = flagged_total / n_windows if n_windows else 0.0
        flagged_windows = [w for w in windows_seen
                           if flagged_by_window.get(w, 0) > 0]
        most_recent = (max(flagged_windows, key=lambda w: WINDOW_ORDER[w])
                       if flagged_windows else None)
        etypes = sorted(set(grp.loc[grp["flagged_events"] > 0, "event_type"].dropna())) \
            if not is_artifact else []
        per_vessel.append({
            "mmsi": mmsi,
            "is_known_artifact": is_artifact,
            "total_windows_observed": n_windows,
            "appears_in_windows": ";".join(windows_seen),
            "flagged_event_count": flagged_total,
            "flagged_event_types": ";".join(etypes),
            "flags_qatar_control_sep2025": int(flagged_by_window.get("qatar_control_sep2025", 0)),
            "flags_qatar_spoofing_oct2025": int(flagged_by_window.get("qatar_spoofing_oct2025", 0)),
            "flags_hormuz_control_jan2026": int(flagged_by_window.get("hormuz_control_jan2026", 0)),
            "flags_hormuz_crisis_mar2026": int(flagged_by_window.get("hormuz_crisis_mar2026", 0)),
            "flag_rate": round(flag_rate, 3),
            "most_recent_flag_window": most_recent,
        })
    return pd.DataFrame(per_vessel)


# ==========================================================================
# 3. GFW vessel-identity cross-reference
# ==========================================================================
def _load_cache():
    p = Path(GFW_CACHE_PATH)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache(cache):
    Path(GFW_CACHE_PATH).write_text(json.dumps(cache, indent=1, sort_keys=True))


def _load_age_cache():
    """MMSI(str) -> {imo, built_year, source, ...}. Manually-curated fallback
    build years for when GFW's registry builtYear is null (which is every
    vessel in this fleet). See the module docstring for why this cannot be a
    fleet-wide automated lookup."""
    p = Path(AGE_CACHE_PATH)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _extract_identity(payload):
    entries = payload.get("entries") or []
    if not entries:
        return {"gfw_identity_available": False}
    e = entries[0]
    reg = e.get("registryInfo") or []
    srp = e.get("selfReportedInfo") or []
    owners = e.get("registryOwners") or []

    flag = next((r["flag"] for r in reg if r.get("flag")), None) \
        or next((r["flag"] for r in srp if r.get("flag")), None)
    imo = next((r["imo"] for r in (reg + srp) if r.get("imo")), None)

    def _scalar(v):
        # GFW extraFields values are sometimes {"value": x, "dateFrom": ...}
        if isinstance(v, dict):
            v = v.get("value")
        return v

    built = None
    for r in reg:
        for ef in (r.get("extraFields") or []):
            raw = _scalar(ef.get("builtYear"))
            try:
                if raw not in (None, ""):
                    built = int(raw)
                    break
            except (TypeError, ValueError):
                pass
        if built:
            break
    iuu = None
    for r in reg:
        for ef in (r.get("extraFields") or []):
            v = (ef.get("iuuStatus") or {}).get("value")
            if v:
                iuu = v
    length = next((r.get("lengthM") for r in reg if r.get("lengthM")), None)
    tonnage = next((r.get("tonnageGt") for r in reg if r.get("tonnageGt")), None)

    # Vessel type: GFW spreads it across registryInfo.geartypes (list of
    # strings) and combinedSourcesInfo[].shiptypes/geartypes (list of dicts
    # with .name). All are coarse ("OTHER"/"CARGO"/"TANKER"); we still record
    # the best available.
    def _typenames(v):
        out = []
        for x in (v or []):
            if isinstance(x, dict) and x.get("name"):
                out.append(str(x["name"]))
            elif isinstance(x, str) and x:
                out.append(x)
        return out

    gear = []
    for r in reg:
        gear += _typenames(r.get("geartypes"))
    ship = []
    for csi in (e.get("combinedSourcesInfo") or []):
        ship += _typenames(csi.get("shiptypes"))
        gear += _typenames(csi.get("geartypes"))

    def _first_specific(vals):
        for v in vals:
            if _norm_type(v) not in _VAGUE_TYPES and _norm_type(v) is not None:
                return v
        return vals[0] if vals else None

    return {
        "gfw_identity_available": True,
        "gfw_flag": flag,
        "gfw_imo": imo,
        "gfw_built_year": built,
        "gfw_iuu_status": iuu,
        "gfw_owner": owners[0].get("name") if owners else None,
        "gfw_owner_flag": owners[0].get("flag") if owners else None,
        "gfw_length_m": length,
        "gfw_tonnage_gt": tonnage,
        "gfw_shiptype": _first_specific(ship),
        "gfw_geartype": _first_specific(gear),
    }


def gfw_lookup(mmsi_list, cache, enabled=True, refresh=False, max_lookups=None):
    """Fill `cache` (mmsi str -> identity dict) for the given MMSIs. Never
    raises: on any failure the entry is marked unavailable."""
    if not enabled:
        print("  GFW lookup disabled (--no-gfw): all identity fields left null.")
        return cache
    if not GFW_API_TOKEN:
        print("  GFW_API_TOKEN not set: skipping identity lookup, fields left null.")
        return cache

    def _stale(m):
        ent = cache.get(str(m))
        if ent is None:
            return True
        if ent.get("_error"):
            return True  # previous transient failure -- retry
        # entry predates the vessel-type fields -> refetch to fill them in
        return ent.get("gfw_identity_available") and "gfw_shiptype" not in ent

    todo = [m for m in mmsi_list if refresh or str(m) not in cache or _stale(m)]
    n_cached = len(mmsi_list) - len(todo)
    if max_lookups is not None and len(todo) > max_lookups:
        print(f"  GFW identity: --max-gfw {max_lookups} -> deferring "
              f"{len(todo) - max_lookups} lookups to a later run.")
        todo = todo[:max_lookups]
    if not todo:
        print(f"  GFW identity: all {len(mmsi_list)} target vessels already cached.")
        return cache

    print(f"  GFW identity: fetching {len(todo)} vessels "
          f"({n_cached} already in cache)...")
    sess = requests.Session()
    sess.headers.update({"Authorization": f"Bearer {GFW_API_TOKEN}"})
    url = f"{GFW_API_BASE_URL}/vessels/search"
    n_ok = n_missing = n_err = 0
    for i, mmsi in enumerate(todo, 1):
        params = {
            "query": str(mmsi), "limit": 1,
            "datasets[0]": "public-global-vessel-identity:latest",
            "includes[0]": "OWNERSHIP",
        }
        try:
            r = None
            for attempt in range(4):  # retry transient network / rate-limit
                try:
                    r = sess.get(url, params=params, timeout=25)
                except (requests.ConnectionError, requests.Timeout):
                    if attempt == 3:
                        raise
                    time.sleep(2 * (attempt + 1))
                    continue
                if r.status_code == 429:
                    time.sleep(3 * (attempt + 1))
                    continue
                break
            r.raise_for_status()
            ident = _extract_identity(r.json())
            cache[str(mmsi)] = ident
            if ident.get("gfw_identity_available"):
                n_ok += 1
            else:
                n_missing += 1
            time.sleep(0.1)  # be polite; the earlier burst got us rate-limited
        except Exception as exc:  # network, JSON, or unexpected payload shape
            prev = cache.get(str(mmsi))
            if prev and prev.get("gfw_identity_available") and not prev.get("_error"):
                # never discard a previously-good record because of a
                # transient failure on a re-fetch -- keep it, note the miss
                prev["_refetch_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
            else:
                cache[str(mmsi)] = {
                    "gfw_identity_available": False,
                    "_error": f"{type(exc).__name__}: {str(exc)[:180]}"}
            n_err += 1
        if i % 25 == 0:
            print(f"    {i}/{len(todo)} (ok={n_ok} missing={n_missing} err={n_err})")
            _save_cache(cache)
    _save_cache(cache)
    print(f"  GFW identity done: matched={n_ok}, no-record={n_missing}, errors={n_err}")
    return cache


# ==========================================================================
# 4. Scoring
# ==========================================================================
def _csv_flag_map():
    """MMSI -> flag state from the CSVs that already carry it (no API call)."""
    fmap = {}
    for path in [f"{PROC}/qatar_control_sept2025_classified.csv",
                 f"{PROC}/hormuz_crisis_mar2026_reclassified.csv",
                 f"{PROC}/hormuz_control_mar2026_reclassified.csv"]:
        df = pd.read_csv(path)
        if "flag" not in df.columns:
            continue
        key = "mmsi"
        m = _num_mmsi(df[key])
        if "matched_mmsi" in df.columns:
            m = m.fillna(_num_mmsi(df["matched_mmsi"]))
        sub = pd.DataFrame({"m": m, "flag": df["flag"]}).dropna(subset=["m"])
        for mmsi, fl in sub.groupby("m")["flag"].agg(
                lambda s: s.dropna().iloc[0] if s.notna().any() else None).items():
            fmap.setdefault(int(mmsi), fl)
    return fmap


def _csv_type_map():
    """MMSI -> vessel_type from the CSVs that carry it (coarse GFW 4wings
    buckets: CARGO / OTHER_NON_FISHING / PASSENGER / GEAR / ...)."""
    tmap = {}
    for path in [f"{PROC}/qatar_control_sept2025_classified.csv",
                 f"{PROC}/hormuz_crisis_mar2026_reclassified.csv",
                 f"{PROC}/hormuz_control_mar2026_reclassified.csv"]:
        df = pd.read_csv(path)
        if "vessel_type" not in df.columns:
            continue
        m = _num_mmsi(df["mmsi"])
        if "matched_mmsi" in df.columns:
            m = m.fillna(_num_mmsi(df["matched_mmsi"]))
        sub = pd.DataFrame({"m": m, "t": df["vessel_type"]}).dropna(subset=["m"])
        for mmsi, tv in sub.groupby("m")["t"].agg(
                lambda s: s.dropna().iloc[0] if s.notna().any() else None).items():
            tmap.setdefault(int(mmsi), tv)
    return tmap


def score_vessels(vessels, cache, names, csv_flags, csv_types, age_cache=None):
    age_cache = age_cache or {}
    out = vessels.copy()
    out["ship_name"] = out["mmsi"].map(names)

    ident = out["mmsi"].map(lambda m: cache.get(str(m), {}))
    out["gfw_identity_available"] = ident.map(lambda d: bool(d.get("gfw_identity_available")))
    out["gfw_built_year"] = ident.map(lambda d: d.get("gfw_built_year"))
    out["gfw_owner"] = ident.map(lambda d: d.get("gfw_owner"))
    out["gfw_owner_flag"] = ident.map(lambda d: d.get("gfw_owner_flag"))
    out["gfw_imo"] = ident.map(lambda d: d.get("gfw_imo"))
    out["gfw_iuu_status"] = ident.map(lambda d: d.get("gfw_iuu_status"))
    out["gfw_shiptype"] = ident.map(lambda d: d.get("gfw_shiptype"))
    out["gfw_geartype"] = ident.map(lambda d: d.get("gfw_geartype"))

    # --- vessel type resolution: confirmed override > GFW > pipeline CSV ---
    def _resolve_type(mmsi, gfw_ship, gfw_gear):
        if mmsi in CONFIRMED_VESSEL_TYPES:
            return CONFIRMED_VESSEL_TYPES[mmsi], "confirmed_external_lookup"
        for cand in (gfw_ship, gfw_gear):
            n = _norm_type(cand)
            if n is not None and n not in _VAGUE_TYPES:
                return cand, "gfw_identity"
        csv_t = csv_types.get(mmsi)
        if _norm_type(csv_t) is not None and _norm_type(csv_t) not in _VAGUE_TYPES:
            return csv_t, "pipeline_csv"
        # keep a vague type rather than a blank column
        if isinstance(gfw_ship, str) and gfw_ship.strip():
            return gfw_ship, "gfw_identity"
        if isinstance(gfw_gear, str) and gfw_gear.strip():
            return gfw_gear, "gfw_identity"
        if isinstance(csv_t, str) and csv_t.strip():
            return csv_t, "pipeline_csv"
        return None, "unknown"

    rtypes, rsources = [], []
    for m, gs, gg in zip(out["mmsi"], out["gfw_shiptype"], out["gfw_geartype"]):
        t, s = _resolve_type(m, gs, gg)
        rtypes.append(t)
        rsources.append(s)
    out["resolved_vessel_type"] = rtypes
    out["vessel_type_source"] = rsources
    out["vessel_type_caveat"] = out["resolved_vessel_type"].map(_is_small_utility)

    gfw_flag = ident.map(lambda d: d.get("gfw_flag"))
    csv_flag = out["mmsi"].map(csv_flags)
    out["flag_state"] = gfw_flag.fillna(csv_flag)
    out["flag_source"] = np.where(gfw_flag.notna(), "gfw_identity",
                          np.where(csv_flag.notna(), "pipeline_csv", "unknown"))

    # --- component 1: behavioural ---
    out["behavioral_score"] = (
        W_BEHAVIORAL * np.minimum(1.0, out["flag_rate"] / FLAG_RATE_CAP)
    ).round(1)

    # --- component 2: flag of convenience (v3 two-tier) ---
    # v2 gave a flat +20 to ANY flag in ITF_FOC_FLAGS | SHADOW_FLEET_FLAGS.
    # That does not discriminate risk: Liberia / Panama / Marshall Islands /
    # Malta between them carry a huge share of ALL legitimate world tonnage,
    # so the component was effectively firing on "large commercial vessel"
    # (13 of the v2 top-14 matched via the broad ITF list; PATRIS -- an LNG
    # carrier managed by K Line and externally confirmed legitimate -- scored
    # the same +20 as a genuine shadow-fleet match). v3 splits it:
    #   shadow_fleet  -> +W_FOC_SHADOW (20)   flag in SHADOW_FLEET_FLAGS, the
    #                    narrow cited 2023-2026 dark-fleet reflagging list.
    #   itf_foc_only  -> +W_FOC_ITF_ONLY (5)  flag in ITF_FOC_FLAGS and NOT in
    #                    SHADOW_FLEET_FLAGS -- real but weak / non-discriminating.
    #   none          -> 0
    # foc_list keeps the v2 tag string ("ITF_FOC", "shadow_fleet", or both)
    # for continuity; foc_list_matched is the new authoritative tier column.
    def foc_list(fl):
        if not isinstance(fl, str):
            return ""
        tags = []
        if fl in ITF_FOC_FLAGS:
            tags.append("ITF_FOC")
        if fl in SHADOW_FLEET_FLAGS:
            tags.append("shadow_fleet")
        return ";".join(tags)
    out["foc_list"] = out["flag_state"].map(foc_list)

    def _foc_tier(fl):
        if isinstance(fl, str) and fl in SHADOW_FLEET_FLAGS:
            return "shadow_fleet"
        if isinstance(fl, str) and fl in ITF_FOC_FLAGS:
            return "itf_foc_only"
        return "none"
    out["foc_list_matched"] = out["flag_state"].map(_foc_tier)
    out["foc_match"] = out["foc_list_matched"] != "none"
    out["foc_score"] = np.select(
        [out["foc_list_matched"].to_numpy() == "shadow_fleet",
         out["foc_list_matched"].to_numpy() == "itf_foc_only"],
        [W_FOC_SHADOW, W_FOC_ITF_ONLY],
        default=0.0,
    )

    # --- component 3: vessel age (v4: GFW builtYear -> age-cache fallback) ---
    # GFW's registry builtYear is null for every vessel in this fleet, so v1-v3
    # scored age 0 across the board. v4 falls back to vessel_age_cache.json
    # (manually resolved by IMO/MMSI from public registry pages) when GFW is
    # null. resolved_imo does the same for the IMO column. built_year_source
    # records provenance so the fallback is never silent.
    gfw_by = pd.to_numeric(out["gfw_built_year"], errors="coerce")

    def _ac(m):
        return age_cache.get(str(int(m)), {}) if pd.notna(m) else {}

    cache_by = out["mmsi"].map(lambda m: _ac(m).get("built_year"))
    cache_by = pd.to_numeric(cache_by, errors="coerce")
    cache_imo = out["mmsi"].map(lambda m: _ac(m).get("imo"))

    out["resolved_built_year"] = gfw_by.where(gfw_by.notna(), cache_by)
    out["built_year_source"] = np.where(
        gfw_by.notna(), "gfw",
        np.where(cache_by.notna(), "age_cache", "none"))
    out["resolved_imo"] = out["gfw_imo"].where(out["gfw_imo"].notna(), cache_imo)

    age = CURRENT_YEAR - out["resolved_built_year"]
    out["vessel_age_years"] = age
    out["age_over_15yr"] = np.where(age.notna(), age > AGE_THRESHOLD_YEARS, None)
    out["age_score"] = np.where(age.notna() & (age > AGE_THRESHOLD_YEARS), W_AGE, 0.0)
    out["age_known"] = age.notna()

    # --- v2: small-utility-craft behavioural down-weight ---
    out["behavioral_multiplier"] = np.where(
        out["vessel_type_caveat"], UTILITY_BEHAVIORAL_MULTIPLIER, 1.0)
    out["behavioral_score_adjusted"] = (
        out["behavioral_score"] * out["behavioral_multiplier"]).round(1)

    # --- totals ---
    # reliability_score_no_caveat == the v1 formula, kept for direct comparison
    out["reliability_score_no_caveat"] = (out["behavioral_score"]
                                          + out["foc_score"] + out["age_score"]).round(1)
    # reliability_score == v2: behavioural component uses the adjusted value
    out["reliability_score"] = (out["behavioral_score_adjusted"]
                                + out["foc_score"] + out["age_score"]).round(1)

    # Confirmed artifacts contribute nothing to any score component -- their
    # flagged history is already zeroed; here we also suppress the static
    # penalties so their row cannot read as "concern".
    art = out["is_known_artifact"]
    for c in ("behavioral_score", "behavioral_score_adjusted", "foc_score",
              "age_score", "reliability_score", "reliability_score_no_caveat"):
        out.loc[art, c] = 0.0
    out.loc[art, "foc_match"] = False

    out = out.sort_values(
        ["reliability_score", "flag_rate", "flagged_event_count"],
        ascending=False,
    ).reset_index(drop=True)
    return out


# ==========================================================================
# 5-6. Report + save
# ==========================================================================
COLUMN_ORDER = [
    "mmsi", "ship_name", "is_known_artifact",
    "total_windows_observed", "appears_in_windows",
    "flagged_event_count", "flagged_event_types",
    "flags_qatar_control_sep2025", "flags_qatar_spoofing_oct2025",
    "flags_hormuz_control_jan2026", "flags_hormuz_crisis_mar2026",
    "flag_rate", "most_recent_flag_window",
    "gfw_identity_available", "flag_state", "flag_source",
    "resolved_vessel_type", "vessel_type_source", "vessel_type_caveat",
    "gfw_shiptype", "gfw_geartype",
    "foc_match", "foc_list", "foc_list_matched",
    "gfw_built_year", "resolved_built_year", "built_year_source",
    "vessel_age_years", "age_over_15yr",
    "gfw_owner", "gfw_owner_flag", "gfw_imo", "resolved_imo", "gfw_iuu_status",
    "behavioral_score", "behavioral_multiplier", "behavioral_score_adjusted",
    "foc_score", "age_score",
    "reliability_score_no_caveat", "reliability_score",
]


def _s(v, dash="?"):
    """Safe string for possibly-NaN/None cell values."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return dash
    return str(v)


def _fmt_row(r):
    name = _s(r["ship_name"], "?")[:18]
    vt = _s(r["resolved_vessel_type"], "type?")[:16]
    cav = " *CAVEAT*" if r["vessel_type_caveat"] else ""
    beh = (f"beh={r['behavioral_score_adjusted']:4.1f}"
           + (f"(raw {r['behavioral_score']:.0f})" if r["vessel_type_caveat"] else ""))
    return (f"  {r['reliability_score']:5.1f}  {str(int(r['mmsi'])):>9}  {name:<18} "
            f"{vt:<16}{cav:9} {beh} foc={r['foc_score']:4.1f} age={r['age_score']:4.1f} "
            f"| rate={r['flag_rate']:.2f} "
            f"({r['flagged_event_count']} in {r['total_windows_observed']}w) "
            f"flag={_s(r['flag_state'])}({_s(r['foc_list_matched'], '-')}) "
            f"recent={_s(r['most_recent_flag_window'], '-')}")


def report(scored):
    print(f"\n{'='*110}\nPER-VESSEL RELIABILITY v3  (0-100, HIGHER = LESS RELIABLE / MORE CONCERN)")
    print(f"{'='*110}")
    print(f"  score model:  behavioral(0-60) = 60*min(1, flag_rate/{FLAG_RATE_CAP:g})   "
          f"+ foc(shadow={int(W_FOC_SHADOW)} / itf_only={int(W_FOC_ITF_ONLY)} / none=0)   "
          f"+ age>15yr(0/{int(W_AGE)})")
    print(f"  flag_rate = flagged_event_count / total_windows_observed  "
          f"(normalises 1-window vs 4-window vessels)")
    print(f"  v2 change: for small utility craft the behavioural component is "
          f"x{UTILITY_BEHAVIORAL_MULTIPLIER:g} (foc/age unchanged).")
    print(f"  v3 change: FOC component is two-tier. A flag in the narrow "
          f"SHADOW_FLEET_FLAGS list keeps +{int(W_FOC_SHADOW)}; a flag in the")
    print(f"    broad ITF_FOC_FLAGS list ONLY (not shadow-fleet) drops to "
          f"+{int(W_FOC_ITF_ONLY)} -- real but weak, since most legitimate")
    print(f"    large commercial tonnage flies those flags. New column: "
          f"foc_list_matched (shadow_fleet / itf_foc_only / none).")
    print(f"\n  SHADOW_FLEET_FLAGS (+{int(W_FOC_SHADOW)}) -- registries repeatedly named in "
          f"2023-2026 dark-fleet / sanctions-evasion")
    print(f"    reporting (S&P Global, Lloyd's List, KSE Institute, UANI):")
    print(f"    {', '.join(sorted(SHADOW_FLEET_FLAGS))}")
    print(f"  ITF_FOC_FLAGS only (+{int(W_FOC_ITF_ONLY)}) -- ITF Fair Practices Committee declared "
          f"flags of convenience (labour/tax):")
    print(f"    {', '.join(sorted(ITF_FOC_FLAGS - SHADOW_FLEET_FLAGS))}")

    # --- explicit vessel-type category + confirmed overrides (printed) ---
    print(f"\n  SMALL-UTILITY-CRAFT category (behavioural weight x"
          f"{UTILITY_BEHAVIORAL_MULTIPLIER:g}); matched on normalised type or keyword:")
    print("    types   : " + ", ".join(sorted(SMALL_UTILITY_CRAFT_TYPES)))
    print("    keywords: " + ", ".join(_UTILITY_KEYWORDS))
    print("    NOT matched: OTHER / OTHER_NON_FISHING / CARGO / TANKER / PASSENGER / "
          "GEAR (too broad -- would hide real concerns)")
    print("  CONFIRMED vessel types (manual external registry lookups this session, "
          "override coarse API/CSV type):")
    for m, t in CONFIRMED_VESSEL_TYPES.items():
        nm = scored.loc[scored["mmsi"] == m, "ship_name"]
        print(f"    {m}  {(_s(nm.iloc[0]) if len(nm) else '?'):<14} -> {t}")

    # --- data-coverage caveat (affects how to read everything below) ---
    wc = scored["total_windows_observed"].value_counts().sort_index()
    win_flag_totals = {w: int(scored[f"flags_{w}"].sum()) for w in WINDOW_ORDER}
    print(f"\n  DATA COVERAGE:")
    print(f"    vessels by #windows observed: "
          + ", ".join(f"{k}w:{v}" for k, v in wc.items()))
    print(f"    flagged events by window: " + ", ".join(
        f"{WINDOW_LABEL[w]}={n}" for w, n in win_flag_totals.items()))
    print("    The two Qatar windows contribute almost nothing: those outputs "
          "predate the likely_spoofed")
    print("    reclassification and routed far-off detections into an "
          "'unmatched' bucket that carries NO MMSI,")
    print("    so Qatar spoofing cannot be keyed to vessels here. Cross-window "
          "history is therefore mostly")
    print("    Hormuz-Jan vs Hormuz-Mar; treat single-window scores as "
          "provisional.")

    ranked = scored[~scored["is_known_artifact"]]
    print(f"\n  TOP 15 of {len(ranked)} scored vessels "
          f"({(ranked['reliability_score'] > 0).sum()} have score > 0):")
    for _, r in ranked.head(15).iterrows():
        print(_fmt_row(r))

    # --- the payoff: vessels flagged in MORE THAN ONE window ---
    flag_cols = [f"flags_{w}" for w in WINDOW_ORDER]
    n_flag_windows = (scored[flag_cols] > 0).sum(axis=1)
    cross = scored[(n_flag_windows >= 2) & ~scored["is_known_artifact"]] \
        .sort_values("reliability_score", ascending=False)
    print(f"\n  CROSS-WINDOW FLAGGED VESSELS ({len(cross)}) -- history only visible "
          "after consolidation:")
    if cross.empty:
        print("    (none -- no vessel is flagged in more than one window)")
    for _, r in cross.iterrows():
        pw = ", ".join(f"{WINDOW_LABEL[w].split(' (')[0]}={int(r[f'flags_{w}'])}"
                       for w in WINDOW_ORDER if r[f"flags_{w}"] > 0)
        tag = (f"  [{_s(r['resolved_vessel_type'])} -- CAVEAT, v1 would be "
               f"{r['reliability_score_no_caveat']:.1f}]" if r["vessel_type_caveat"] else "")
        print(f"    {r['reliability_score']:5.1f}  {_s(r['ship_name']):<20} "
              f"({int(r['mmsi'])})  rate={r['flag_rate']:.2f}  [{pw}]{tag}")

    # --- v2: small-utility-craft caveat analysis (requirement 4) ---
    print(f"\n{'-'*110}\n  SMALL-UTILITY-CRAFT CAVEAT -- effect on scores")
    print(f"{'-'*110}")
    caveated = scored[scored["vessel_type_caveat"] & ~scored["is_known_artifact"]] \
        .sort_values("reliability_score_no_caveat", ascending=False)
    by_src = caveated["vessel_type_source"].value_counts().to_dict()
    print(f"  {len(caveated)} vessel(s) matched the small-utility-craft category "
          f"(by source: {by_src}):")
    for _, r in caveated.iterrows():
        d = r["reliability_score"] - r["reliability_score_no_caveat"]
        print(f"    {int(r['mmsi'])}  {_s(r['ship_name']):<14} "
              f"type={_s(r['resolved_vessel_type'])} (src {r['vessel_type_source']})")
        print(f"        reliability_score  {r['reliability_score_no_caveat']:.1f} (v1 formula) "
              f"-> {r['reliability_score']:.1f} (v2)   delta {d:+.1f}")
        print(f"        behavioural  {r['behavioral_score']:.1f} raw -> "
              f"{r['behavioral_score_adjusted']:.1f} (x{UTILITY_BEHAVIORAL_MULTIPLIER:g}); "
              f"foc {r['foc_score']:.0f}, age {r['age_score']:.0f} unchanged   "
              f"(flag_rate {r['flag_rate']:.2f}, {r['flagged_event_count']} events / "
              f"{r['total_windows_observed']} windows)")

    # Did any OTHER top-15 vessel (by either ranking) turn out to be utility?
    nonart = scored[~scored["is_known_artifact"]]
    top_v1 = set(nonart.sort_values("reliability_score_no_caveat", ascending=False)
                 .head(15)["mmsi"])
    top_v2 = set(nonart.head(15)["mmsi"])  # already sorted by v2 score
    other = [int(m) for m in (top_v1 | top_v2)
             if m not in CONFIRMED_VESSEL_TYPES
             and bool(scored.loc[scored["mmsi"] == m, "vessel_type_caveat"].iloc[0])]
    if other:
        print(f"\n  Top-15 vessels (v1 or v2 ranking) that ALSO resolve to small "
              f"utility craft: {other}")
    else:
        print(f"\n  No top-15 vessel (v1 or v2 ranking) resolves to a small utility "
              "craft -- the top of the board is genuine large-vessel behaviour.")
    # but flag utility craft that WERE in the pre-caveat cross-window / concern set
    cross_util = caveated[(caveated["total_windows_observed"] >= 2)
                          | (caveated["reliability_score_no_caveat"] >= 25)]
    if len(cross_util):
        print("  Utility craft that scored as a cross-window / mid-board concern "
              "under v1 and are now down-weighted:")
        for _, r in cross_util.iterrows():
            print(f"    {int(r['mmsi'])} {_s(r['ship_name']):<16} "
                  f"{_s(r['resolved_vessel_type']):<20} "
                  f"v1 {r['reliability_score_no_caveat']:.1f} -> v2 {r['reliability_score']:.1f}")
    print("  GFW `geartypes` resolves TUG / SUPPLY_VESSEL / DREDGE_NON_FISHING "
          "directly; the coarse OTHER / CARGO / TANKER / PASSENGER / GEAR buckets")
    print("  still cannot be checked, so more utility craft may sit unlabelled "
          "among CARGO/OTHER-typed vessels -- those need per-vessel registry lookups.")

    print(f"\n  Excluded artifacts (flagged events zeroed, not scored):")
    for _, r in scored[scored["is_known_artifact"]].iterrows():
        seen = r["appears_in_windows"].replace("hormuz_", "").replace("_", " ")
        print(f"    {int(r['mmsi'])}  {_s(r['ship_name']):<12} observed in [{seen}] "
              f"-- reliability_score forced to {r['reliability_score']:.1f}")

    # --- requirement 5: named vessels + normalisation commentary ---
    print(f"\n{'-'*96}\n  NAMED VESSELS -- single-window vs normalised reading")
    print(f"{'-'*96}")
    for mmsi, nm in NAMED_VESSELS.items():
        row = scored[scored["mmsi"] == mmsi]
        if row.empty:
            print(f"  {nm} ({mmsi}): not present in any window's saved output.")
            continue
        r = row.iloc[0]
        rank_txt = f"#{list(ranked['mmsi']).index(mmsi) + 1}/{len(ranked)}" \
            if mmsi in list(ranked["mmsi"]) else "unranked"
        per_win = {w: int(r[f"flags_{w}"]) for w in WINDOW_ORDER
                   if r[f"flags_{w}"] > 0}
        print(f"\n  {nm} ({mmsi})  reliability_score={r['reliability_score']:.1f}  "
              f"[{rank_txt} among non-artifacts]")
        print(f"    components : behavioral={r['behavioral_score']:.1f}  "
              f"foc={r['foc_score']:.1f} ({_s(r['flag_state'])}, {_s(r['foc_list'], 'not FOC') or 'not FOC'})  "
              f"age={r['age_score']:.1f} "
              f"({'age ' + str(int(r['vessel_age_years'])) + 'y' if pd.notna(r['vessel_age_years']) else 'build year unknown'})")
        print(f"    history    : {r['flagged_event_count']} flagged event(s) across "
              f"{r['total_windows_observed']} observed window(s) -> flag_rate "
              f"{r['flag_rate']:.2f}")
        print(f"    per window : {per_win or 'no flagged events'}")
        print(f"    recency    : most recent flag in "
              f"{WINDOW_LABEL.get(r['most_recent_flag_window'], '-')}")
        # normalisation note
        nwin = r["total_windows_observed"]
        if r["flagged_event_count"] == 0:
            note = ("no flagged events at all -- a clean record under this "
                    "consolidation regardless of any single map's impression.")
        elif nwin <= 1:
            note = (f"all flags come from ONE window -- flag_rate {r['flag_rate']:.2f} "
                    "looks high but rests on a single observation; treat as "
                    "lower-confidence than a vessel flagged repeatedly across windows.")
        elif len(per_win) == 1:
            note = (f"flagged in only 1 of {nwin} observed windows -- normalisation "
                    "pulls it DOWN relative to a raw single-window view: an isolated "
                    "incident, not a persistent pattern.")
        else:
            note = (f"flagged in {len(per_win)} separate windows -- the pattern "
                    "survives normalisation and is a genuine cross-window signal, "
                    "not a one-window artifact.")
        print(f"    read       : {note}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-gfw", action="store_true", help="skip GFW identity lookups")
    ap.add_argument("--refresh-gfw", action="store_true", help="ignore identity cache")
    ap.add_argument("--max-gfw", type=int, default=None,
                    help="cap number of live GFW lookups this run")
    args = ap.parse_args()

    print("Consolidating flagged events across 4 windows ...")
    events = load_events()
    print(f"  {len(events)} (vessel, window, source) observation rows; "
          f"{events['mmsi'].nunique()} distinct vessels; "
          f"{int(events['flagged_events'].sum())} flagged events total")
    vessels = consolidate(events)

    # GFW identity: look up every vessel with a real flagged event, plus the
    # named ones. (Vessels with zero flags get behavioral 0 anyway; skipping
    # them keeps the API load bounded -- documented limitation.)
    to_lookup = sorted(set(
        vessels.loc[(vessels["flagged_event_count"] > 0)
                    & ~vessels["is_known_artifact"], "mmsi"].tolist()
    ) | set(NAMED_VESSELS))
    cache = {} if args.refresh_gfw else _load_cache()
    cache = gfw_lookup(to_lookup, cache, enabled=not args.no_gfw,
                       refresh=args.refresh_gfw, max_lookups=args.max_gfw)
    age_cache = _load_age_cache()

    scored = score_vessels(vessels, cache, _name_map(),
                           _csv_flag_map(), _csv_type_map(), age_cache=age_cache)

    Path(PROC).mkdir(parents=True, exist_ok=True)
    scored[COLUMN_ORDER].to_csv(OUTPUT_FILE_V6, index=False)
    report(scored)

    n_gfw = int(scored["gfw_identity_available"].sum())
    n_age = int(scored["age_known"].sum())
    n_foc = int(scored["foc_match"].sum())
    n_type = int(scored["resolved_vessel_type"].notna().sum())
    n_specific = int((scored["vessel_type_source"].isin(
        ["confirmed_external_lookup", "gfw_identity"])
        & scored["resolved_vessel_type"].map(
            lambda t: _norm_type(t) not in _VAGUE_TYPES)).sum())
    n_cav = int(scored["vessel_type_caveat"].sum())
    print(f"\n  GFW identity resolved for {n_gfw}/{len(scored)} vessels "
          f"({len(scored) - n_gfw} left null -- unflagged vessels are not looked "
          "up, plus a few API misses).")
    n_shadow = int((scored["foc_list_matched"] == "shadow_fleet").sum())
    n_itf_only = int((scored["foc_list_matched"] == "itf_foc_only").sum())
    n_age_gfw = int((scored["built_year_source"] == "gfw").sum())
    n_age_cache = int((scored["built_year_source"] == "age_cache").sum())
    n_imo = int(scored["resolved_imo"].notna().sum())
    print(f"  Static factors: {n_foc} vessels carry a flag-of-convenience flag "
          f"({n_shadow} shadow_fleet +{int(W_FOC_SHADOW)}, {n_itf_only} itf_foc_only "
          f"+{int(W_FOC_ITF_ONLY)}).")
    print(f"  Build year: {n_age} of {n_imo} vessels-with-an-IMO resolved "
          f"({n_age_gfw} from GFW, {n_age_cache} from vessel_age_cache.json). "
          f"{int((scored['age_score'] > 0).sum())} score the age>15yr +{int(W_AGE)}.")
    if n_age == 0:
        print("    -> the age>15yr component contributed 0 to every score this run.")
    print(f"  Vessel type: {n_type} vessels have some type string, but only "
          f"~{n_specific} are more specific than a broad bucket; "
          f"{n_cav} matched the small-utility-craft caveat.")
    print(f"  v1..v5 files left untouched.")
    print(f"  v6 table ({len(scored)} vessels, FINAL) -> {OUTPUT_FILE_V6}")


if __name__ == "__main__":
    main()
