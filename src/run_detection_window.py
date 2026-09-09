"""Run the full detection chain on ONE arbitrary window, standalone.

This does not touch the hard-coded 4-window historical scorer
(src.vessel_history) or the dashboard. It reuses the shipped core functions
(fetch -> relink -> match -> trajectory -> ST-DBSCAN -> watchlist tier-1/2)
and writes per-window outputs under the given label:

    data/raw/<label>_sar_detections.csv
    data/raw/<label>_ais_positions.csv
    data/processed/<label>_relinked.csv
    data/processed/<label>_classified.csv
    data/processed/<label>_trajectory_deviation.csv
    data/processed/<label>_spatiotemporal_clusters.csv

then runs watchlist_trigger.run() against <label> with its tier-1 source
swapped to those files (tier-2 deep-history gate / cache log / audit trail
are reused verbatim).

Usage (from a small caller, see __main__):
    from src.run_detection_window import run_window
    run_window("india_mangalore_dec2025_jan2026", BBOX,
               "2025-12-01", "2026-01-15",
               matched_km=2.6434, spoofed_km=20.2469,
               eps_km=8.9773, eps_h=6.0)
"""
import numpy as np
import pandas as pd

PROC = "data/processed"
RAW = "data/raw"


def _fetch(label, bbox, start, end):
    from src.fetch_sar import fetch_sar_detections
    from src.fetch_ais import fetch_ais_positions

    sar = fetch_sar_detections(bbox, start, end)
    ais = fetch_ais_positions(bbox, start, end)
    sar_path = f"{RAW}/{label}_sar_detections.csv"
    ais_path = f"{RAW}/{label}_ais_positions.csv"
    sar.to_csv(sar_path, index=False)
    ais.to_csv(ais_path, index=False)
    passes = sorted(sar["timestamp"].dt.floor("min").unique()) if len(sar) else []
    print(f"[fetch] {label}: SAR={len(sar)} rows over {len(passes)} pass(es) "
          f"{[str(p)[:16] for p in passes]}; "
          f"AIS={len(ais)} presence bins, {ais['mmsi'].nunique()} MMSI")
    return sar, ais


def _relink(label):
    """verify-then-repair mode, as shipped (Hungarian re-link over each
    overpass; GFW links that pass the mis-attribution screen are locked)."""
    from src import relink_sar_ais as rl

    df, sar, ais, meta = rl.relink_dataset(label, mode="repair")
    out = f"{PROC}/{label}_relinked.csv"
    df.to_csv(out, index=False)
    vc = df["link_status"].value_counts() if "link_status" in df.columns else pd.Series(dtype=int)
    print(f"[relink] {label}: {len(df)} detections over {meta['n_overpasses']} overpass(es); "
          f"{meta['n_locked']} GFW link(s) locked (passed screen). "
          f"status: " + ", ".join(f"{k}={int(v)}" for k, v in vc.items()) + f" -> {out}")
    return df


def _match(label, sar, ais, matched_km, spoofed_km):
    from src.match import match_and_classify

    res = match_and_classify(sar, ais,
                             matched_threshold_km=matched_km,
                             spoofed_threshold_km=spoofed_km)
    out = f"{PROC}/{label}_classified.csv"
    res.to_csv(out, index=False)
    vc = res["classification"].value_counts()
    print(f"[match] {label}: thresholds matched<={matched_km} km / "
          f"likely_spoofed>{spoofed_km} km")
    for k in ("matched", "discrepant", "likely_spoofed", "no_ais_activity"):
        print(f"         {k:16} {int(vc.get(k, 0))}")
    return res


def _trajectory(label, ais):
    from src.trajectory_predict import predict_deviations

    traj = predict_deviations(ais).assign(window=label)
    out = f"{PROC}/{label}_trajectory_deviation.csv"
    traj.to_csv(out, index=False)
    vc = traj["tier"].value_counts()
    print(f"[trajectory] {label}: {len(traj)} predicted points, "
          f"tiers normal={int(vc.get('normal', 0))} "
          f"notable={int(vc.get('notable', 0))} high={int(vc.get('high', 0))}")
    return traj


def _cluster(label, eps_km, eps_h):
    """Monkeypatch spatiotemporal_cluster's hard-coded window globals, then
    call analyze_window() for this one label and persist ping-level output."""
    from src import spatiotemporal_cluster as sc

    sc.RECLASS_FILES = {label: f"{PROC}/{label}_classified.csv"}
    sc.TRAJ_DEVIATION_FILE = f"{PROC}/{label}_trajectory_deviation.csv"
    sc.TRAJ_WINDOW_LABEL = {label: label}
    sc.OUTPUT_FILE = f"{PROC}/{label}_spatiotemporal_clusters.csv"
    sc.EPS_SPATIAL_KM = eps_km
    sc.EPS_TEMPORAL_HOURS = eps_h

    ep_df, points, summary = sc.analyze_window(label)

    sm = summary.set_index("cluster") if len(summary) else summary
    out = points.copy()
    out["cluster_id"] = out["cluster"].apply(
        lambda c: f"{label}-noise" if c < 0 else f"{label}-{c}")

    def _mp(col, default):
        return out["cluster"].map(
            lambda c: sm.at[c, col] if len(sm) and c in sm.index else default)

    out["cluster_n_distinct_vessels"] = _mp("n_distinct_vessels", 0)
    out["cluster_n_episodes"] = _mp("n_episodes", 0)
    out["cluster_time_span_hours"] = _mp("time_span_hours", np.nan)
    out["cluster_radius_km"] = _mp("radius_km", np.nan)
    out["cluster_coherence_label"] = _mp("coherence_label", "noise")
    out = out.drop(columns=["cluster"])
    out.to_csv(sc.OUTPUT_FILE, index=False)

    n_gen = int((summary["coherence_label"] == "genuine_multivessel_coherence").sum()) \
        if len(summary) else 0
    print(f"[cluster] {label}: eps {eps_km} km / {eps_h} h -> "
          f"{len(summary)} cluster(s), {n_gen} genuine_multivessel_coherence "
          f"-> {sc.OUTPUT_FILE}")
    return out, summary


# --------------------------------------------------------------------------
# watchlist tier-1 shim -- same thresholds/semantics as
# src.watchlist_trigger.flagged_vessels, but sourced from THIS window's files
# --------------------------------------------------------------------------
def _make_flagged_vessels(label):
    from src import vessel_history as vh
    from src import watchlist_trigger as wt

    def flagged_vessels(window_key):
        assert window_key == label, (window_key, label)
        names = {}
        out, artifacts = {}, {}

        def _rec(mmsi):
            return out.setdefault(int(mmsi), {"ship_name": names.get(int(mmsi)),
                                             "thresholds": set(),
                                             "event_counts": {}})

        # -- SAR-vs-AIS likely_spoofed --
        cls = pd.read_csv(f"{PROC}/{label}_classified.csv")
        for _, r in cls.iterrows():
            if r.get("ship_name") == r.get("ship_name") and pd.notna(r.get("ship_name")):
                for k in ("mmsi", "matched_mmsi"):
                    if pd.notna(r.get(k)):
                        names.setdefault(int(r[k]), r["ship_name"])
        vessel = pd.to_numeric(cls["mmsi"], errors="coerce")
        if "matched_mmsi" in cls.columns:
            vessel = vessel.fillna(pd.to_numeric(cls["matched_mmsi"], errors="coerce"))
        cls = cls.assign(_v=vessel)
        spoof = cls[(cls["classification"].astype(str) == "likely_spoofed")
                    & cls["_v"].notna()]
        for mmsi, g in spoof.groupby("_v"):
            n = vh._count_episodes(g["sar_timestamp"])
            if n:
                rec = _rec(mmsi)
                rec["thresholds"].add("sar_ais_likely_spoofed")
                rec["event_counts"]["sar_ais_likely_spoofed"] = \
                    rec["event_counts"].get("sar_ais_likely_spoofed", 0) + int(n)

        # -- trajectory high tier --
        tj = pd.read_csv(f"{PROC}/{label}_trajectory_deviation.csv")
        tj = tj[(tj["tier"] == "high") & tj["mmsi"].notna()]
        for mmsi, g in tj.groupby("mmsi"):
            names.setdefault(int(mmsi), g["ship_name"].dropna().iloc[0]
                             if g["ship_name"].notna().any() else None)
            n = vh._count_episodes(g["predicted_timestamp"])
            if n:
                rec = _rec(mmsi)
                rec["thresholds"].add("trajectory_deviation_high")
                rec["event_counts"]["trajectory_deviation_high"] = \
                    rec["event_counts"].get("trajectory_deviation_high", 0) + int(n)

        # -- genuine multi-vessel cluster membership --
        scv = pd.read_csv(f"{PROC}/{label}_spatiotemporal_clusters.csv")
        key = "vessel_mmsi" if "vessel_mmsi" in scv.columns else "mmsi"
        gen = scv[scv["cluster_coherence_label"] == "genuine_multivessel_coherence"]
        for mmsi, g in gen.groupby(key):
            if pd.isna(mmsi):
                continue
            n = int(g["cluster_id"].nunique())
            if n:
                rec = _rec(mmsi)
                rec["thresholds"].add("genuine_cluster_member")
                rec["event_counts"]["genuine_cluster_member"] = \
                    rec["event_counts"].get("genuine_cluster_member", 0) + n

        # finalise: sort thresholds to label strings, split known artifacts
        final, arts = {}, {}
        for mmsi, rec in out.items():
            rec["ship_name"] = names.get(mmsi)
            rec["thresholds"] = sorted(
                wt.THRESHOLD_OF.get(t, t) for t in rec["thresholds"])
            (arts if mmsi in vh.EXCLUDE_MMSIS else final)[mmsi] = rec
        return final, arts

    return flagged_vessels


def _watchlist(label, max_pulls, dry_run, no_gfw):
    from src import watchlist_trigger as wt

    wt.flagged_vessels = _make_flagged_vessels(label)  # swap tier-1 source only
    return wt.run(label, dry_run=dry_run, max_pulls=max_pulls, no_gfw=no_gfw)


FISHING_NAME_RE = r"(?i)\b(net|boat|buoy|bouy|msv|dol)\b|net[- ]?\d|_\d{1,3}%|%$"


def _merchant_ais(ais, keep_vessel_types):
    """Restrict the AIS pool feeding trajectory + clustering to the merchant
    fleet. GFW buckets a lot of real cargo/tankers as OTHER, so OTHER is kept
    unless the ship name looks like an artisanal fishing / net-buoy unit."""
    if keep_vessel_types is None:
        return ais
    vt = ais["vessel_type"].astype(str)
    keep = vt.isin(keep_vessel_types)
    if "OTHER" in keep_vessel_types:
        nm = ais["ship_name"].astype(str)
        keep = keep & ~(vt.eq("OTHER") & nm.str.contains(FISHING_NAME_RE, regex=True, na=False))
    out = ais[keep].copy()
    print(f"[filter] merchant-only AIS: {len(ais)} -> {len(out)} bins, "
          f"{ais['mmsi'].nunique()} -> {out['mmsi'].nunique()} MMSI "
          f"(kept {sorted(keep_vessel_types)})")
    return out


def run_window(label, bbox, start, end, *, matched_km, spoofed_km,
               eps_km, eps_h, max_pulls=8, dry_run=False, no_gfw=False,
               skip_fetch=False, keep_vessel_types=None, skip_watchlist=False):
    print(f"\n{'#'*96}\n# DETECTION WINDOW  {label}  |  {start} .. {end}  |  bbox {bbox}\n{'#'*96}")
    if skip_fetch:
        sar = pd.read_csv(f"{RAW}/{label}_sar_detections.csv", parse_dates=["timestamp"])
        ais = pd.read_csv(f"{RAW}/{label}_ais_positions.csv", parse_dates=["timestamp"])
        print(f"[fetch] skipped -- using cached raw ({len(sar)} SAR / {len(ais)} AIS rows)")
    else:
        sar, ais = _fetch(label, bbox, start, end)

    if len(sar) == 0:
        print("[abort] no SAR detections in this window -- nothing to detect on.")
        return None

    _relink(label)
    _match(label, sar, ais, matched_km, spoofed_km)          # classification: full AIS pool
    ais_beh = _merchant_ais(ais, keep_vessel_types)           # behaviour stages: merchant only
    _trajectory(label, ais_beh)
    _cluster(label, eps_km, eps_h)
    res = None
    if skip_watchlist:
        print("[watchlist] skipped (skip_watchlist=True)")
    else:
        res = _watchlist(label, max_pulls, dry_run, no_gfw)
    print(f"\n[done] {label}")
    return res
