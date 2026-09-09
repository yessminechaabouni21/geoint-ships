# Maritime-Cyber GEOINT: AIS/GNSS Spoofing Detection

Detecting GNSS/AIS position spoofing in shadow-fleet shipping through AIS-SAR
cross-verification, trajectory self-consistency, and multi-vessel spatiotemporal
clustering — validated against real, documented incidents in the Persian Gulf
and Indian waters.

> A ship broadcasting as **"LENORE"** sat frozen at one coordinate for four
> straight days, then jumped 152 km in a single hour — an implied 82 knots,
> faster than any commercial vessel can travel. LENORE is a false identity for
> **DAVINA**, a tanker sanctioned by the United States in 2024 for shipping
> Iranian crude in violation of sanctions. This pipeline is what caught it.

---

## What This Is

GNSS/AIS spoofing — ships broadcasting a false position — is a core technique
used to hide sanctions-evading "shadow fleet" oil shipments. This project
cross-checks what a vessel *claims* about its own position against:

1. **Independent SAR satellite detections** (does the satellite see it where it says it is?)
2. **Its own trajectory** (is its movement physically consistent with its own recent speed/heading?)
3. **Other nearby vessels** (do multiple ships show correlated anomalies, or is this one vessel in isolation?)
4. **Its own dwell/loiter behavior** (is it suspiciously motionless at a flagged facility?)

Every vessel that clears a screening threshold gets escalated to a deep,
6-month individual history pull and external registry verification before
being treated as a real finding — not just a flagged number.

## Key Findings

| Finding | Detail |
|---|---|
| **A data-source flaw, not just a spoofing detector** | Global Fishing Watch's own SAR-to-AIS linking mis-attributes vessels in dense coastal traffic (5/5 confirmed cases in Indian waters, ~30–100 km errors). Fixed with a Hungarian-algorithm (Kuhn-Munkres) re-linking layer — classical optimization, not ML. |
| **A quantified detection floor** | At subtle spoofing magnitudes (2–5 nautical miles, the documented Oct 2025 Qatar incident), hourly-bin AIS resolution cannot separate signal from noise. This is a real, tested limitation, not an assumption. |
| **A structural blind spot found and closed** | Two named, independently-confirmed shadow-fleet tankers (TIBURON, SEASONS I) at a sanctioned Indian refinery triggered *zero* existing checks — because the entire architecture was built to catch anomalous movement, and their suspicious behavior was sitting motionless for 9+ days. A dwell-time detector was built, validated, and closes this gap with zero false positives on ordinary port queuing. |
| **LENORE / DAVINA** | Four independent lines of evidence: sanctioned real identity, a physically impossible 82-knot jump, a flagged staging-area location (Larak Island), and a deviation from its own 6-month baseline. Explicitly caveated as a strong single-vessel candidate, not a confirmed multi-vessel-corroborated event (per ST-DBSCAN clustering check). |
| **A self-correcting scoring system** | The vessel reliability score (0–100) was corrected three times after real external verification caught its own biases — e.g. a harbor tug scoring as a top suspect until its type was confirmed and down-weighted. Every vessel in the final top-15 is externally audited against real ship registries. |

## Architecture

```
                    TIER 1 — BROAD SCREENING (cheap, every vessel)
   ┌──────────────────────────────────────────────────────────────┐
   │  AIS pull → SAR pull + re-linking → AIS-vs-SAR match          │
   │  → trajectory self-consistency → ST-DBSCAN clustering         │
   │  → loiter/dwell detection                                      │
   └──────────────────────────────┬───────────────────────────────┘
                                   │  flagged vessels only
                                   ▼
                    TIER 2 — DEEP INVESTIGATION (expensive, targeted)
   ┌──────────────────────────────────────────────────────────────┐
   │  6-month per-vessel history → external registry verification  │
   │  → composite reliability score → 30-day cache                 │
   └────────────────────────────────────────────────────────────────┘
```

## Repository Structure

```
src/
  fetch_sar.py             Pull SAR vessel detections (GFW/Sentinel-1)
  fetch_ais.py              Pull AIS hourly-bin positions (GFW)
  relink_sar_ais.py          Hungarian-algorithm SAR-AIS re-linking (verify-then-repair)
  interpolate.py             Bracket-interpolate AIS positions to SAR timestamps
  match.py                   Classify: matched / discrepant / likely_spoofed / no_ais_activity
  trajectory_predict.py       Per-vessel self-consistency (constant-velocity model)
  route_plausibility.py       Sea-route + real-elapsed-time plausibility check
  spatiotemporal_cluster.py   ST-DBSCAN multi-vessel coherence (Park et al., 2026)
  loiter_detector.py          Dwell/loiter detection near flagged facilities
  calibrate_thresholds.py     Region-agnostic automatic threshold calibration
  vessel_history.py           Composite reliability scoring (behavioral + flag + age)
  vessel_deep_history.py      6-month per-vessel deep history pull
  watchlist_trigger.py        Two-tier auto-escalation with 30-day caching
  dashboard.py                Interactive Streamlit + pydeck dashboard
  generate_figures.py         Static explainable figures (matplotlib)

data/
  raw/                       Raw AIS/SAR pulls (gitignored, regenerable)
  processed/                 Classified results, scores, calibration JSON (tracked)

figures/                    Generated PNG figures for presentation/report
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your own API keys
```

Requires:
- **Global Fishing Watch API key** — free registration at
  [globalfishingwatch.org/our-apis](https://globalfishingwatch.org/our-apis)
  (state research/non-commercial purpose)
- **AISStream API key** — free at [aisstream.io](https://aisstream.io)

## Usage

```bash
# 1. Calibrate thresholds for a new region (run once, cache reused after)
python -m src.calibrate_thresholds <region_label> --fetch

# 2. Run the full detection chain for a window
python -m src.run_detection_window <window_label>

# 3. Screen and auto-escalate flagged vessels to deep investigation
python -m src.watchlist_trigger --window <window_label> --dry-run   # preview
python -m src.watchlist_trigger --window <window_label> --max-pulls 8

# 4. Launch the interactive dashboard
streamlit run src/dashboard.py
```

## Data Sources

| Source | Used for | Access |
|---|---|---|
| Global Fishing Watch API | AIS positions, SAR vessel detections, vessel identity | Free, registered |
| AISStream | Live AIS feed | Free, registered |
| Equasis / vessel registries | Build year, ownership (external verification) | Free |

**Key limitation:** AIS data is hourly grid-cell presence (~1.1 km resolution),
not raw pings. This sets a hard precision floor, documented and tested
throughout this project rather than assumed away.

## Validated Regions/Windows

| Region | Windows | Role |
|---|---|---|
| Qatar / Ras Laffan | Oct 2025 incident + Sep 2025 control | Detection floor (negative result) |
| Strait of Hormuz | Mar 2026 crisis + Jan 2026 control | Severe-spoofing validation; LENORE |
| New Mangalore, India | Baseline + detection windows | India transfer validation (clean) |
| Jamnagar/Vadinar, India | Baseline + Feb/May 2026 detection | Named shadow-fleet vessel follow-up |

## Methodology

This project explicitly evaluated and **rejected** deep learning and quantum
ML for its core detection tasks — the coarse AIS data resolution is the
binding constraint, not model capacity. Every threshold and algorithm choice
is documented with its reasoning in code comments. See
`Phase1_AIS_SAR_Methodology_and_Findings.md` for the full narrative writeup,
including every bug found, tested, and fixed.

## Citations

- Park et al. (2026). AIS-based GNSS RFI monitoring via ST-DBSCAN. arXiv:2603.11055
- Nguyen et al. GeoTrackNet: a-contrario anomaly detection for maritime traffic
- Paolo et al. (2022). xView3-SAR. arXiv:2206.00897
- Louart et al. (2024). AIS identity spoofing detection via carrier frequency offset

## Limitations

- Single-vessel anomalies (e.g. LENORE) are unconfirmed without multi-vessel
  ST-DBSCAN corroboration — treated as strong candidates, not proof
- Dashboard is ships-only by design (planes/cables/floods deliberately deferred)
- Sentinel-1 revisit frequency limits sample density per detection window

## Repository Access

Kept private with collaborators added directly — GFW API redistribution
terms for derived data have not been independently verified for public
release.
