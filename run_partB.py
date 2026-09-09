from src.run_detection_window import run_window

# Jamnagar refinery waters + Vadinar terminal, Gulf of Kutch.
BBOX = {"min_lat": 22.2, "max_lat": 22.7, "min_lon": 69.5, "max_lon": 70.1}

# Jamnagar/Vadinar's OWN calibrated thresholds
# (threshold_calibration_jamnagar_vadinar.json -> derived_thresholds):
#   matched_max 0.0 km  /  likely_spoofed_min 7.4632 km
#   ST-DBSCAN eps 3.0 km / 2.0 h
# Detection window: 2026-05-05..2026-05-26 -- best-sampled recent window
# (2 Sentinel-1 passes May 12 + May 24, 150 SAR detections, 84% AIS-matched).
# No merchant filter: Vadinar is a deep-water terminal with negligible
# artisanal fishing fleet (unlike New Mangalore).
run_window(
    "jamnagar_vadinar_may2026", BBOX,
    "2026-05-05", "2026-05-26",
    matched_km=0.0, spoofed_km=7.4632,
    eps_km=3.0, eps_h=2.0,
    max_pulls=8,
)
