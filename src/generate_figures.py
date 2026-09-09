"""Generate the 5 presentation figures (PNG, matplotlib only).

For an ML / cybersec engineering audience: each figure is meant to stand on
its own without narration -- consistent style, labelled axes, direct
annotations. All numeric content is pulled from the real project artefacts
(the raw AIS track, the reliability-score CSV, the re-linking validation
results); nothing is placeholder.

Run:  python -m src.generate_figures
Output: figures/*.png  (folder created if missing)
"""
import os
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIG_DIR = "figures"
RAW = "data/raw"
PROC = "data/processed"

# ----------------------------------------------------------------------
# Shared style
# ----------------------------------------------------------------------
PALETTE = {
    "ink":       "#232323",
    "muted":     "#6b7280",
    "grid":      "#e6e6e6",
    "tier1":     "#2a9d8f",   # teal  -- cheap / broad
    "tier2":     "#e76f51",   # orange -- expensive / narrow
    "skip":      "#9aa4ab",   # grey  -- the no-op cache path
    "bad":       "#c1121f",   # raw / anomaly
    "neutral":   "#264653",
    "good":      "#2a9d8f",
    "warn":      "#e29500",
    "behavioral":"#264653",
    "foc":       "#e9c46a",
    "age":       "#8ab17d",
}
FIGSIZE = (16, 10)   # ~1700 x 1150 px after tight-bbox crop at dpi=120
DPI = 120


def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "text.color": PALETTE["ink"],
        "axes.edgecolor": "#cccccc",
        "axes.labelcolor": PALETTE["ink"],
        "axes.titlecolor": PALETTE["ink"],
        "axes.titlesize": 17,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 12,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": True,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.8,
    })


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.35, facecolor="white")
    plt.close(fig)
    print(f"  saved  {path}")
    return path


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ======================================================================
# 1. LENORE implied-speed anomaly (line chart, real hourly track)
# ======================================================================
def fig_lenore_speed():
    src = f"{RAW}/hormuz_crisis_mar2026_ais_positions.csv"
    d = pd.read_csv(src, parse_dates=["timestamp"])
    lo_t = pd.Timestamp("2026-03-19 00:00", tz="UTC")
    hi_t = pd.Timestamp("2026-03-23 15:00", tz="UTC")
    d = (d[(d["mmsi"] == 306531000) & d["timestamp"].between(lo_t, hi_t)]
         .sort_values("timestamp").reset_index(drop=True))

    lat, lon = d["lat"].to_numpy(float), d["lon"].to_numpy(float)
    ts = pd.DatetimeIndex(d["timestamp"])
    t = ts.tz_convert("UTC").tz_localize(None).to_numpy()   # datetime64, safe for np.diff
    seg_km = _haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
    dt_h = np.diff(t) / np.timedelta64(1, "h")
    speed_kn = (seg_km / dt_h) / 1.852          # km/h -> knots
    t_mid = ts[1:]                                # tz-aware DatetimeIndex

    peak_i = int(np.argmax(speed_kn))
    peak_kn = float(speed_kn[peak_i])
    peak_t = pd.Timestamp(t_mid[peak_i])
    frozen_end = peak_t - pd.Timedelta(hours=1)
    frozen_start = pd.Timestamp(t_mid[0])
    frozen_days = (frozen_end - frozen_start) / pd.Timedelta(days=1)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    # frozen span
    ax.axvspan(frozen_start, frozen_end, color=PALETTE["tier1"], alpha=0.10, zorder=0)
    ax.text(frozen_start + (frozen_end - frozen_start) * 0.42, 52,
            f"position frozen for ≈{frozen_days:.1f} days\n"
            f"(hourly step ≤ 0.54 kn → grid jitter only)",
            ha="center", va="center", fontsize=12, color=PALETTE["neutral"],
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=PALETTE["tier1"], lw=1))

    # normal-transit reference band
    ax.axhspan(7, 13, color=PALETTE["good"], alpha=0.10, zorder=0)
    ax.text(pd.Timestamp(t_mid[-1]), 10, "  normal transit 7–13 kn",
            va="center", ha="left", fontsize=11, color=PALETTE["good"])

    ax.plot(t_mid, speed_kn, color=PALETTE["neutral"], lw=1.8, zorder=3)
    ax.scatter([peak_t], [peak_kn], s=90, color=PALETTE["bad"], zorder=5,
               edgecolor="white", linewidth=1.2)

    ax.annotate(
        f"{peak_kn:.0f} kn — physically impossible\n"
        f"({peak_t:%b %d %H:%M}: single 1-hour step of ~152 km)",
        xy=(peak_t, peak_kn), xytext=(peak_t - pd.Timedelta(hours=34), peak_kn - 6),
        ha="right", va="center", fontsize=12.5, fontweight="bold", color=PALETTE["bad"],
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["bad"], lw=1.8,
                        connectionstyle="arc3,rad=-0.15"))

    ax.annotate("then decays back to\nnormal 7–13 kn transit",
                xy=(pd.Timestamp(t_mid[peak_i + 4]), float(speed_kn[peak_i + 4])),
                xytext=(pd.Timestamp(t_mid[peak_i + 4]) + pd.Timedelta(hours=3), 40),
                ha="left", fontsize=11, color=PALETTE["muted"],
                arrowprops=dict(arrowstyle="-|>", color=PALETTE["muted"], lw=1.3))

    ax.set_title("LENORE (MMSI 306531000): implied speed from its own hourly AIS track "
                 "— Hormuz crisis window")
    ax.set_xlabel("UTC")
    ax.set_ylabel("implied speed between consecutive hourly fixes  (knots)")
    ax.set_ylim(-4, 90)
    ax.margins(x=0.02)
    fig.autofmt_xdate()
    ax.text(0.005, -0.13,
            f"source: {src}  ·  speed = haversine(fixₙ₋₁, fixₙ) / 1 h.  "
            "Frozen position then a one-sample jump is the classic AIS position-spoof signature.",
            transform=ax.transAxes, fontsize=9.5, color=PALETTE["muted"])
    return save(fig, "lenore_speed_anomaly.png")


# ======================================================================
# 2. likely_spoofed threshold by re-linking method (bar chart)
# ======================================================================
def fig_threshold_comparison():
    # Values from the India / New Mangalore SAR<->AIS re-linking validation
    # (src/relink_sar_ais.py STEP 5 table).
    labels = ["Raw GFW\n(trust the given MMSI)",
              "Screen + exclude\nbad pairs",
              "Verify-then-repair\n(shipped)",
              "Blind full\nre-solve"]
    values = [78.78, 20.25, 20.22, 9.90]
    colors = [PALETTE["bad"], PALETTE["neutral"], PALETTE["good"], PALETTE["tier2"]]

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    x = np.arange(len(labels))
    bars = ax.bar(x, values, width=0.6, color=colors, edgecolor="white", linewidth=1.5)

    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.4, f"{v:.2f} km",
                ha="center", va="bottom", fontsize=13, fontweight="bold")

    # match annotation between screen-only and shipped
    ax.plot([1, 2], [26, 26], color=PALETTE["muted"], lw=1.2)
    ax.plot([1, 1], [22, 26], color=PALETTE["muted"], lw=1.2)
    ax.plot([2, 2], [22, 26], color=PALETTE["muted"], lw=1.2)
    ax.text(1.5, 28,
            "shipped method lands on the independent\nscreen-only result "
            "(20.22 vs 20.25 km) — two paths, same answer",
            ha="center", va="bottom", fontsize=11.5, color=PALETTE["neutral"])

    ax.annotate("over-corrected —\nlinks each detection to\nits nearest boat",
                xy=(3, 9.9), xytext=(3, 44), ha="center", fontsize=11.5,
                color=PALETTE["tier2"],
                arrowprops=dict(arrowstyle="-|>", color=PALETTE["tier2"], lw=1.5))

    ax.annotate("one 97 km GFW mis-attribution\nalone sets this cutoff",
                xy=(0.3, 74), xytext=(0.5, 60), ha="left", fontsize=11.5,
                color=PALETTE["bad"],
                arrowprops=dict(arrowstyle="-|>", color=PALETTE["bad"], lw=1.5))

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylabel("derived  likely_spoofed  threshold  (km)")
    ax.set_ylim(0, 90)
    ax.set_title("SAR↔AIS re-linking: derived likely_spoofed threshold by "
                 "verification method  (India / New Mangalore)")
    ax.grid(axis="x", visible=False)
    ax.text(0.005, -0.12,
            "India / New Mangalore baseline, 155 confirmed identity pairs.  "
            "matched-tier cutoff is unaffected (~2.6 km) in every method.",
            transform=ax.transAxes, fontsize=9.5, color=PALETTE["muted"])
    return save(fig, "threshold_correction_comparison.png")


# ======================================================================
# 3. Two-tier architecture (boxes + arrows)
# ======================================================================
def _box(ax, cx, cy, w, h, text, fc, tc="white", fs=11.5, weight="normal"):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                linewidth=0, facecolor=fc, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", color=tc,
            fontsize=fs, fontweight=weight, zorder=3, linespacing=1.35)
    return (cx, cy)


def _arrow(ax, p0, p1, color="#5b6670"):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.7,
                                shrinkA=6, shrinkB=6), zorder=1)


def fig_two_tier():
    fig, ax = plt.subplots(figsize=(16, 11), dpi=DPI)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # tier bands
    ax.add_patch(FancyBboxPatch((0.15, 5.15), 9.7, 4.7,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                fc=PALETTE["tier1"], alpha=0.09, linewidth=0))
    ax.add_patch(FancyBboxPatch((0.15, 0.15), 9.7, 4.75,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                fc=PALETTE["tier2"], alpha=0.09, linewidth=0))
    ax.text(0.35, 9.55, "TIER 1 — broad screening", fontsize=14,
            fontweight="bold", color=PALETTE["tier1"])
    ax.text(0.35, 9.2, "cheap · runs on every vessel, every window",
            fontsize=10.5, color=PALETTE["tier1"])
    ax.text(0.35, 4.62, "TIER 2 — deep investigation", fontsize=14,
            fontweight="bold", color=PALETTE["tier2"])
    ax.text(0.35, 4.28, "expensive · GFW deep pulls · flagged vessels only",
            fontsize=10.5, color="#b5482f")

    t1, t2 = PALETTE["tier1"], PALETTE["tier2"]
    pull = _box(ax, 5.0, 8.6, 5.2, 0.95,
                "AIS + SAR pull\nfetch_ais.py   ·   fetch_sar.py", t1, weight="bold")
    det_y = 6.9
    d1 = _box(ax, 2.15, det_y, 2.7, 1.35,
              "match.py\nSAR↔AIS distance tiers\nmatched / discrepant / spoofed", t1, fs=10.5)
    d2 = _box(ax, 5.0, det_y, 2.7, 1.35,
              "trajectory_predict.py\nconstant-velocity\ndeviation tiers", t1, fs=10.5)
    d3 = _box(ax, 7.85, det_y, 2.7, 1.35,
              "spatiotemporal_cluster.py\nST-DBSCAN\nmulti-vessel coincidence", t1, fs=10.5)
    flagged = _box(ax, 5.0, 5.5, 4.4, 0.8, "flagged vessels", PALETTE["neutral"],
                   weight="bold")
    for d in (d1, d2, d3):
        _arrow(ax, (pull[0], pull[1] - 0.48), (d[0], d[1] + 0.68))
        _arrow(ax, (d[0], d[1] - 0.68), (flagged[0], flagged[1] + 0.4))

    wl = _box(ax, 5.0, 3.9, 5.6, 0.95,
              "watchlist_trigger.py\n30-day cache check (per-vessel TTL)", t2, weight="bold")
    _arrow(ax, (flagged[0], flagged[1] - 0.4), (wl[0], wl[1] + 0.48), color=t2)

    fresh = _box(ax, 3.05, 2.25, 3.6, 1.25,
                 "FRESH_PULL\nvessel_deep_history.py\n6-month GFW history", t2, fs=10.5)
    skip = _box(ax, 7.15, 2.25, 3.6, 1.25,
                "CACHE_SKIP\nwithin 30-day TTL\n(no API spend)", PALETTE["skip"], fs=10.5)
    _arrow(ax, (wl[0] - 1.2, wl[1] - 0.48), (fresh[0], fresh[1] + 0.63), color=t2)
    _arrow(ax, (wl[0] + 1.2, wl[1] - 0.48), (skip[0], skip[1] + 0.63), color=PALETTE["skip"])

    score = _box(ax, 5.0, 0.75, 5.6, 0.9,
                 "vessel_history.py\nreliability scoring  (v1 → v3)", t2, weight="bold")
    _arrow(ax, (fresh[0], fresh[1] - 0.63), (score[0] - 0.9, score[1] + 0.45), color=t2)
    _arrow(ax, (skip[0], skip[1] - 0.63), (score[0] + 0.9, score[1] + 0.45), color=PALETTE["skip"])

    ax.set_title("Real-time architecture: broad Tier-1 screening feeds narrow, "
                 "expensive Tier-2 investigation", pad=18)
    return save(fig, "two_tier_architecture.png")


# ======================================================================
# 4. Bugs found & fixed (horizontal timeline)
# ======================================================================
def fig_bugs_timeline():
    bugs = [
        ("Wrong timestamp field (Qatar)",
         "Positioned on entryTimestamp/exitTimestamp (= query-range bounds), so "
         "every grid cell \"contained\" every query time.",
         "fetch_ais.py"),
        ("Spoofing hidden in \"unmatched\"",
         "A capped nearest-AIS search dumped far-away activity into an unmatched "
         "bucket — severe spoofing read as AIS silence.",
         "match.py"),
        ("Vessel-type bias in scoring",
         "Score leaned on vessel type, penalising whole categories instead of "
         "observed behaviour.",
         "vessel_history.py  (v2)"),
        ("FOC list over-broadness",
         "Flag-of-convenience matching was too permissive, inflating the FOC "
         "component for ordinary flags.",
         "vessel_history.py  (v3)"),
        ("Misleading deviation_km on long gaps",
         "A fixed km cutoff over a 20 h AIS gap flagged ordinary route flex; "
         "moved to sqrt(gap) scaling.",
         "route_plausibility.py"),
        ("GFW SAR↔AIS mis-attribution",
         "GFW's given MMSI was wrong by tens of km in dense coastal traffic; "
         "added verify-then-repair re-linking.",
         "relink_sar_ais.py"),
    ]
    fig, ax = plt.subplots(figsize=(16, 9), dpi=DPI)
    n = len(bugs)
    xs = np.arange(1, n + 1)
    ax.plot([0.5, n + 0.5], [0, 0], color=PALETTE["neutral"], lw=2.5, zorder=1)
    ax.scatter(xs, np.zeros(n), s=140, color=PALETTE["neutral"], zorder=3,
               edgecolor="white", linewidth=1.5)

    for i, (x, (title, desc, fname)) in enumerate(zip(xs, bugs)):
        above = (i % 2 == 0)
        stub = 0.40 if above else -0.40
        anchor = 0.52 if above else -0.52
        va = "bottom" if above else "top"
        ax.plot([x, x], [0, stub], color=PALETTE["muted"], lw=1.2, zorder=2)
        ec = PALETTE["tier2"] if i == n - 1 else "#cfd4d8"
        head = f"{i + 1}.  {title}"
        body = head + "\n" + textwrap.fill(desc, 44) + f"\n— {fname}"
        ax.text(x, anchor, body, ha="center", va=va, fontsize=10.3,
                linespacing=1.5, color=PALETTE["ink"],
                bbox=dict(boxstyle="round,pad=0.6", fc="white", ec=ec, lw=1.3))

    ax.set_xlim(0.15, n + 0.85)
    ax.set_ylim(-1.35, 1.35)
    ax.axis("off")
    ax.set_title("Bugs found & fixed, in order of discovery  "
                 "(each traced to one file)", pad=16)
    ax.text(0.5, -1.3, "direction of project development →",
            fontsize=10, color=PALETTE["muted"])
    return save(fig, "bugs_timeline.png")


# ======================================================================
# 5. Reliability-score breakdown (stacked bar, v3 values)
# ======================================================================
def fig_score_breakdown():
    src = f"{PROC}/vessel_reliability_scores_v6.csv"
    df = pd.read_csv(src)
    want = ["SELENIA", "PATRIS", "OCEAN CENTURY"]
    df = df[df["ship_name"].isin(want)].set_index("ship_name").loc[want]

    beh = df["behavioral_score_adjusted"].to_numpy(float)
    beh_raw = df["behavioral_score"].to_numpy(float)
    mult = df["behavioral_multiplier"].to_numpy(float)
    foc = df["foc_score"].to_numpy(float)
    age = df["age_score"].to_numpy(float)
    total = df["reliability_score"].to_numpy(float)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    x = np.arange(len(want))
    w = 0.5
    b1 = ax.bar(x, beh, w, label="behavioral", color=PALETTE["behavioral"],
                edgecolor="white", linewidth=1.2)
    b2 = ax.bar(x, foc, w, bottom=beh, label="flag-of-convenience (FOC)",
                color=PALETTE["foc"], edgecolor="white", linewidth=1.2)
    b3 = ax.bar(x, age, w, bottom=beh + foc, label="vessel age",
                color=PALETTE["age"], edgecolor="white", linewidth=1.2)

    ages = df["vessel_age_years"].to_numpy(float)
    for xi, (be, br, mu, fo, ag, agey, tot) in enumerate(
            zip(beh, beh_raw, mult, foc, age, ages, total)):
        ax.text(xi, tot + 1.6, f"{tot:.0f}", ha="center", va="bottom",
                fontsize=15, fontweight="bold")
        ax.text(xi, be / 2, f"{be:.0f}", ha="center", va="center",
                color="white", fontsize=12, fontweight="bold")
        if fo > 0:
            ax.text(xi, be + fo / 2, f"{fo:.0f}", ha="center", va="center",
                    color=PALETTE["ink"], fontsize=11, fontweight="bold")
        if ag > 0:
            ax.text(xi, be + fo + ag / 2, f"{ag:.0f}", ha="center", va="center",
                    color="white", fontsize=11, fontweight="bold")
        if mu < 1.0:
            ax.annotate(
                f"confirmed tug → v2 small-utility-craft\n"
                f"down-weight:  behavioral {br:.0f} × {mu:g} = {be:.0f}",
                xy=(xi, be), xytext=(xi, 50),
                ha="center", va="center", fontsize=10, color=PALETTE["bad"],
                arrowprops=dict(arrowstyle="-|>", color=PALETTE["bad"], lw=1.4),
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          ec=PALETTE["bad"], lw=1.1))

    def _blt(agey):
        return (f"built {2026 - int(agey)}, {int(agey)} yr"
                if not np.isnan(agey) else "build year n/a")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n({m})   {_blt(a)}"
                        for s, m, a in zip(want, df["mmsi"], ages)])
    ax.set_ylabel("reliability score  (0–100, higher = more concerning)")
    ax.set_ylim(0, 112)
    ax.set_xlim(-0.62, len(want) - 0.38)
    ax.set_title("Reliability score breakdown (v6): behavioral + FOC + age components")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="x", visible=False)
    cap = (
        f"source: {src}.  build years from vessel_age_cache.json (GFW builtYear null for this whole fleet).\n"
        "All three flagged 4× in one CRISIS window (raw behavioral 60).  FOC separates SELENIA (shadow-fleet flag,\n"
        "+20) from PATRIS / OCEAN CENTURY (ITF-FOC only, +5).  age>15 yr adds +20 to SELENIA (22 yr) and OCEAN\n"
        "CENTURY (19 yr); PATRIS (8 yr) is genuinely young.  OCEAN CENTURY is a confirmed tug — its behavioral\n"
        "component takes the v2 ×0.25 small-utility-craft down-weight (60→15), dropping it from #2 to #37 fleet-wide."
    )
    ax.text(0.005, -0.30, cap, transform=ax.transAxes, fontsize=9.5,
            color=PALETTE["muted"], linespacing=1.5)
    return save(fig, "vessel_score_breakdown.png")


# ======================================================================
def main():
    apply_style()
    print("Generating figures ->", os.path.abspath(FIG_DIR))
    fig_lenore_speed()
    fig_threshold_comparison()
    fig_two_tier()
    fig_bugs_timeline()
    fig_score_breakdown()
    print("done — 5 figures written.")


if __name__ == "__main__":
    main()
