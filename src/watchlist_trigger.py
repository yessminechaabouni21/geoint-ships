"""Two-tier real-time escalation, wired into one automatic chain.

TIER 1 -- broad, cheap, every vessel.  The existing screening chain already
writes per-window outputs:
    match.py / reclassify_hormuz.py  -> SAR-vs-AIS classification
    trajectory_predict.py            -> deviation tiers
    spatiotemporal_cluster.py        -> cluster coherence
This module consumes those outputs through src.vessel_history.load_events --
the SAME consolidation the scorer uses, so "flagged here" means exactly what
"flagged" means everywhere else in the project -- and picks out every vessel
in the target window that crossed ANY threshold:
    * SAR-vs-AIS  classification == likely_spoofed
    * trajectory  tier == high
    * cluster     cluster_coherence_label == genuine_multivessel_coherence

TIER 2 -- deep, expensive, flagged vessels only.  src.vessel_deep_history
pulls 6 months of per-vessel GFW event history.  That call is rate-limited
and pointless to repeat on unchanged history, so it is gated by a persistent
cache log (data/processed/deep_history_cache_log.json):

    never deep-pulled, OR last pull > DEEP_HISTORY_TTL_DAYS old
        -> FRESH_PULL  (run vessel_deep_history for this MMSI, bump pull_count)
    pulled within the TTL
        -> CACHE_SKIP  (increment reflag_count instead -- a vessel that keeps
                        re-flagging is itself a signal, recorded without
                        spending an API call)

If the cache log does not exist yet it is BOOTSTRAPPED from any
vessel_deep_history_<mmsi>.csv files already on disk (their mtime becomes the
last-pull time) -- i.e. deep history we already have counts as cached.

Every decision -- flagged / cache-skip / fresh-pull / deferred / error --
is printed and appended to an append-only audit log
(data/processed/watchlist_trigger_log.jsonl), consistent with the project's
explainability standard.

Run:
    python -m src.watchlist_trigger --window hormuz_crisis_mar2026
    python -m src.watchlist_trigger --window hormuz_crisis_mar2026 --dry-run
    python -m src.watchlist_trigger --window hormuz_crisis_mar2026 --max-pulls 5
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src import vessel_history as vh
from src import vessel_deep_history as vdh
from src import loiter_detector as ld

PROC = "data/processed"
CACHE_LOG = f"{PROC}/deep_history_cache_log.json"
AUDIT_LOG = f"{PROC}/watchlist_trigger_log.jsonl"

# Deep history older than this is considered stale and re-pulled; within it,
# a re-flag is logged but no API call is made. Same style of explicit,
# tunable constant as the thresholds in vessel_history / route_plausibility.
DEEP_HISTORY_TTL_DAYS = 30

# Vessels discussed across the project -- reported explicitly in the summary
# so their handling under the automated logic is always visible.
KNOWN_VESSELS = {
    306531000: "LENORE",
    511101414: "SELENIA",
    636018010: "PATRIS",
    636025162: "OCEAN CENTURY",
}

# The tier-1 thresholds, mapped from the event_type strings that
# vessel_history.load_events() -- and src.loiter_detector.load_loiter_events()
# -- emit. `prolonged_dwell_at_flagged_facility` was added after the
# TIBURON / SEASONS I stress test showed a multi-day stationary hold at a
# sanctioned refinery produced ZERO flags from the spoofing / deviation /
# cluster chain (an idle vessel has ~0 km trajectory deviation and never
# enters ST-DBSCAN). See src/loiter_detector.py.
THRESHOLD_OF = {
    "sar_ais_likely_spoofed": "likely_spoofed (SAR-vs-AIS)",
    "trajectory_deviation_high": "high-tier trajectory deviation",
    "genuine_cluster_member": "genuine_multivessel_coherence cluster",
    "prolonged_dwell_at_flagged_facility": "prolonged dwell at flagged facility",
}


def _now():
    return datetime.now(timezone.utc)


def _iso(dt):
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _parse(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


# ==========================================================================
# Tier 1 -- flagged vessels for a window
# ==========================================================================
def flagged_vessels(window_key):
    """{mmsi: {"ship_name", "thresholds": sorted[str], "event_counts": {..}}}
    for every vessel that crossed a tier-1 threshold in `window_key`.
    Known confirmed-artifact MMSIs (vessel_history.EXCLUDE_MMSIS) are
    reported separately and never escalated."""
    events = vh.load_events()
    names = vh._name_map()

    # Additional tier-1 category: prolonged stationary dwell at a flagged
    # facility (src.loiter_detector). Present only for windows that have a
    # data/processed/loiter_flags_{window}.csv on disk; contributes nothing
    # otherwise, so existing windows are unaffected unless a flags file is
    # generated for them.
    loiter = ld.load_loiter_events(window_key)
    if len(loiter):
        events = pd.concat([events, loiter], ignore_index=True)
        names = {**ld.name_map(window_key), **names}

    if window_key not in set(events["window"]):
        raise SystemExit(
            f"no screening events for window '{window_key}'. "
            f"known windows: {sorted(set(events['window']))}")

    sub = events[(events["window"] == window_key) & (events["flagged_events"] > 0)]
    out, artifacts = {}, {}
    for mmsi, g in sub.groupby("mmsi"):
        mmsi = int(mmsi)
        rec = {
            "ship_name": names.get(mmsi),
            "thresholds": sorted({THRESHOLD_OF.get(t, t)
                                  for t in g["event_type"].dropna()}),
            "event_counts": {t: int(g.loc[g["event_type"] == t, "flagged_events"].sum())
                             for t in sorted(g["event_type"].dropna().unique())},
        }
        if mmsi in vh.EXCLUDE_MMSIS:
            artifacts[mmsi] = rec
        else:
            out[mmsi] = rec
    return out, artifacts


# ==========================================================================
# Tier 2 gate -- persistent cache log
# ==========================================================================
def load_cache_log():
    p = Path(CACHE_LOG)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            pass
    return bootstrap_cache_log()


def bootstrap_cache_log():
    """First run: treat every vessel_deep_history_<mmsi>.csv already on disk
    as a prior pull, timed at the file's mtime."""
    log = {}
    for f in sorted(Path(PROC).glob("vessel_deep_history_*.csv")):
        try:
            mmsi = int(f.stem.rsplit("_", 1)[1])
        except (ValueError, IndexError):
            continue
        mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
        log[str(mmsi)] = {
            "ship_name": vdh.SHORTLIST.get(mmsi) or KNOWN_VESSELS.get(mmsi),
            "first_pulled": _iso(mtime), "last_pulled": _iso(mtime),
            "pull_count": 1, "reflag_count": 0, "last_reflag": None,
            "flagged_windows": [], "last_decision": "bootstrap_from_existing_file",
        }
    Path(CACHE_LOG).write_text(json.dumps(log, indent=1, sort_keys=True))
    print(f"  cache log did not exist -- bootstrapped from "
          f"{len(log)} existing deep-history file(s): "
          f"{sorted(int(k) for k in log)}")
    return log


def save_cache_log(log):
    Path(CACHE_LOG).write_text(json.dumps(log, indent=1, sort_keys=True))


def decide(mmsi, log, now):
    """-> (action, reason, age_days | None). action in {FRESH_PULL, CACHE_SKIP}."""
    ent = log.get(str(mmsi))
    if ent is None:
        return "FRESH_PULL", "never deep-pulled", None
    last = _parse(ent.get("last_pulled"))
    if last is None:
        return "FRESH_PULL", "cache entry has no last_pulled timestamp", None
    age_days = (now - last).total_seconds() / 86400.0
    if age_days > DEEP_HISTORY_TTL_DAYS:
        return "FRESH_PULL", f"last pull {age_days:.0f} d ago (> {DEEP_HISTORY_TTL_DAYS} d TTL)", age_days
    return "CACHE_SKIP", f"deep history is {age_days:.1f} d old (<= {DEEP_HISTORY_TTL_DAYS} d TTL)", age_days


# ==========================================================================
# Audit
# ==========================================================================
def audit(record):
    record = {"logged_at": _iso(_now()), **record}
    with open(AUDIT_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


# ==========================================================================
# Chain
# ==========================================================================
def run(window_key, dry_run=False, max_pulls=8, rescore=False, no_gfw=False):
    run_id = _iso(_now())
    print(f"\n{'='*96}")
    print(f"WATCHLIST TRIGGER  |  window={window_key}  |  run={run_id}"
          f"{'  |  DRY RUN' if dry_run else ''}")
    print(f"{'='*96}")

    flagged, artifacts = flagged_vessels(window_key)
    log = load_cache_log()
    now = _now()
    audit({"event": "run_start", "run_id": run_id, "window": window_key,
           "dry_run": dry_run, "n_flagged": len(flagged),
           "n_artifacts_suppressed": len(artifacts),
           "ttl_days": DEEP_HISTORY_TTL_DAYS, "max_pulls": max_pulls})

    print(f"\nTIER 1 -- {len(flagged)} vessel(s) crossed a screening threshold in this window "
          f"({len(artifacts)} known artifact(s) suppressed):")
    for mmsi in sorted(flagged):
        r = flagged[mmsi]
        print(f"  FLAGGED  {mmsi}  {str(r['ship_name'] or '?'):22}  "
              f"via {', '.join(r['thresholds'])}   counts={r['event_counts']}")
        audit({"event": "flagged", "run_id": run_id, "window": window_key,
               "mmsi": mmsi, "ship_name": r["ship_name"],
               "thresholds": r["thresholds"], "event_counts": r["event_counts"]})
    for mmsi, r in sorted(artifacts.items()):
        print(f"  SUPPRESSED (known artifact)  {mmsi}  {str(r['ship_name'] or '?'):22}  "
              f"-- not escalated (vessel_history.EXCLUDE_MMSIS)")
        audit({"event": "artifact_suppressed", "run_id": run_id,
               "window": window_key, "mmsi": mmsi, "ship_name": r["ship_name"]})

    # ---- tier-2 gate ----
    print(f"\nTIER 2 -- escalation decisions (TTL {DEEP_HISTORY_TTL_DAYS} d, "
          f"pull budget this run = {max_pulls}):")
    fresh, skipped, deferred, failed = [], [], [], []
    pulls_done = 0
    for mmsi in sorted(flagged):
        name = flagged[mmsi]["ship_name"] or vdh.SHORTLIST.get(mmsi) or KNOWN_VESSELS.get(mmsi)
        action, reason, age = decide(mmsi, log, now)

        if action == "CACHE_SKIP":
            ent = log[str(mmsi)]
            ent["reflag_count"] = int(ent.get("reflag_count", 0)) + 1
            ent["last_reflag"] = _iso(now)
            ent.setdefault("flagged_windows", [])
            if window_key not in ent["flagged_windows"]:
                ent["flagged_windows"].append(window_key)
            ent["last_decision"] = "cache_skip"
            skipped.append(mmsi)
            print(f"  CACHE_SKIP  {mmsi}  {str(name or '?'):22}  {reason}; "
                  f"re-flag counter -> {ent['reflag_count']}")
            audit({"event": "cache_skip", "run_id": run_id, "window": window_key,
                   "mmsi": mmsi, "ship_name": name, "reason": reason,
                   "age_days": round(age, 2) if age is not None else None,
                   "reflag_count": ent["reflag_count"]})
            continue

        # action == FRESH_PULL
        if pulls_done >= max_pulls and not dry_run:
            deferred.append(mmsi)
            print(f"  DEFERRED    {mmsi}  {str(name or '?'):22}  fresh pull needed "
                  f"({reason}) but pull budget exhausted -- will pull next run")
            audit({"event": "deferred", "run_id": run_id, "window": window_key,
                   "mmsi": mmsi, "ship_name": name, "reason": reason})
            continue

        if dry_run:
            print(f"  FRESH_PULL  {mmsi}  {str(name or '?'):22}  {reason}  [dry-run: not executed]")
            audit({"event": "fresh_pull_planned", "run_id": run_id,
                   "window": window_key, "mmsi": mmsi, "ship_name": name, "reason": reason})
            fresh.append(mmsi)
            continue

        print(f"  FRESH_PULL  {mmsi}  {str(name or '?'):22}  {reason}  -- calling "
              f"vessel_deep_history ...")
        res = vdh.pull_deep_history(mmsi, ship_name=name, enabled=not no_gfw,
                                    do_report=False)
        pulls_done += 1
        if res.get("ok"):
            ent = log.get(str(mmsi), {})
            ent.setdefault("first_pulled", _iso(now))
            ent["ship_name"] = name
            ent["last_pulled"] = _iso(now)
            ent["pull_count"] = int(ent.get("pull_count", 0)) + 1
            ent.setdefault("reflag_count", 0)
            ent.setdefault("flagged_windows", [])
            if window_key not in ent["flagged_windows"]:
                ent["flagged_windows"].append(window_key)
            ent["last_decision"] = "fresh_pull"
            ent["last_pull_n_events"] = res["n_events"]
            ent["last_pull_csv"] = res["csv_path"]
            if res.get("xref"):
                ent["last_pull_reading"] = res["xref"]["better_supported_reading"]
            log[str(mmsi)] = ent
            fresh.append(mmsi)
            rd = res["xref"]["better_supported_reading"] if res.get("xref") else "n/a"
            print(f"              -> {res['n_events']} events -> {res['csv_path']}  "
                  f"(baseline reading: {rd})")
            audit({"event": "fresh_pull", "run_id": run_id, "window": window_key,
                   "mmsi": mmsi, "ship_name": name, "n_events": res["n_events"],
                   "csv_path": res["csv_path"], "pull_count": ent["pull_count"],
                   "baseline_reading": rd})
        else:
            failed.append(mmsi)
            print(f"              -> FAILED: {res.get('error')}  (not marked cached; "
                  f"will retry next run)")
            audit({"event": "fresh_pull_failed", "run_id": run_id,
                   "window": window_key, "mmsi": mmsi, "ship_name": name,
                   "error": res.get("error")})

    if not dry_run:
        save_cache_log(log)

    # ---- optional tier-1 re-score (requirement 4: merge into scoring) ----
    rescored = False
    if rescore and fresh and not dry_run:
        print(f"\n  {len(fresh)} fresh pull(s) -- re-running vessel_history scoring so the "
              f"leaderboard reflects this window ...")
        cp = subprocess.run([sys.executable, "-m", "src.vessel_history", "--no-gfw"],
                            capture_output=True, text=True)
        rescored = cp.returncode == 0
        tail = cp.stdout.strip().splitlines()[-3:] if cp.stdout else []
        for ln in tail:
            print(f"    | {ln}")
        audit({"event": "rescore", "run_id": run_id, "ok": rescored})

    # ---- summary ----
    print(f"\n{'-'*96}\nSUMMARY  (window {window_key})\n{'-'*96}")
    print(f"  vessels flagged (tier 1)      : {len(flagged)}"
          f"   (+{len(artifacts)} known artifacts suppressed)")
    print(f"  fresh deep-history pulls      : {len(fresh)}  {sorted(fresh)}")
    print(f"  cache hits / skipped (re-flag): {len(skipped)}  {sorted(skipped)}")
    if deferred:
        print(f"  deferred (pull budget)       : {len(deferred)}  {sorted(deferred)}")
    if failed:
        print(f"  fresh-pull failures          : {len(failed)}  {sorted(failed)}")
    print(f"  tier-1 re-score run           : {'yes' if rescored else 'no'}")
    print(f"  cache log                     : {CACHE_LOG}")
    print(f"  audit trail                   : {AUDIT_LOG}")

    print(f"\n  KNOWN VESSELS under the automated logic:")
    for mmsi, nm in KNOWN_VESSELS.items():
        if mmsi in flagged:
            if mmsi in fresh:
                tag = "flagged -> FRESH_PULL (no recent deep history)"
            elif mmsi in skipped:
                rc = log.get(str(mmsi), {}).get("reflag_count", "?")
                tag = f"flagged -> CACHE_SKIP (deep history current; re-flag counter = {rc})"
            elif mmsi in deferred:
                tag = "flagged -> FRESH_PULL deferred (pull budget)"
            elif mmsi in failed:
                tag = "flagged -> FRESH_PULL attempted, FAILED"
            else:
                tag = "flagged"
        else:
            tag = "not flagged in this window -- no escalation (correct)"
        print(f"    {mmsi}  {nm:14} : {tag}")

    audit({"event": "run_end", "run_id": run_id, "window": window_key,
           "n_flagged": len(flagged), "n_fresh": len(fresh),
           "n_skipped": len(skipped), "n_deferred": len(deferred),
           "n_failed": len(failed), "rescored": rescored})
    return {"flagged": flagged, "fresh": fresh, "skipped": skipped,
            "deferred": deferred, "failed": failed}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", default="hormuz_crisis_mar2026",
                    help="screening window key (default: hormuz_crisis_mar2026)")
    ap.add_argument("--dry-run", action="store_true",
                    help="decide and log, but execute no deep-history pulls")
    ap.add_argument("--max-pulls", type=int, default=8,
                    help="cap fresh deep-history pulls this run (rest deferred)")
    ap.add_argument("--rescore", action="store_true",
                    help="after fresh pulls, re-run src.vessel_history scoring")
    ap.add_argument("--no-gfw", action="store_true",
                    help="do not hit the GFW API (fresh pulls only from cache)")
    args = ap.parse_args()
    run(args.window, dry_run=args.dry_run, max_pulls=args.max_pulls,
        rescore=args.rescore, no_gfw=args.no_gfw)


if __name__ == "__main__":
    main()
