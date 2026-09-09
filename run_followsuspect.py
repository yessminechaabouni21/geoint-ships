from src.run_detection_window import run_window

# Follow-the-suspect: TIBURON (IMO 9283291, MMSI 667001287) + SEASONS I
# (IMO 9308950, MMSI 613511209), both named by CNBC/Kpler (2026-02-03) at/near
# Nayara's Vadinar refinery. Window brackets Feb 3 with SAR passes Jan 24 +
# Feb 5 (2 passes over Jan 20 - Feb 17).
BBOX = {"min_lat": 22.2, "max_lat": 22.7, "min_lon": 69.5, "max_lon": 70.1}

# Jamnagar/Vadinar's own PROVISIONAL calibrated thresholds
# (threshold_calibration_jamnagar_vadinar.json): matched_max 0.0 km /
# likely_spoofed_min 7.4632 km, ST-DBSCAN eps 3.0 km / 2.0 h. Confidence LOW.
run_window(
    "jamnagar_vadinar_feb2026", BBOX,
    "2026-01-20", "2026-02-17",
    matched_km=0.0, spoofed_km=7.4632,
    eps_km=3.0, eps_h=2.0,
    skip_watchlist=True,  # the tier-1 flood makes auto-escalation meaningless here (see Part B);
                          # the 2 named vessels are extracted + deep-pulled explicitly afterwards
)
