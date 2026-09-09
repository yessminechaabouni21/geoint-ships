from src.run_detection_window import run_window

# New Mangalore approaches + outer anchorages (same bbox as calibration).
BBOX = {"min_lat": 12.0, "max_lat": 13.8, "min_lon": 73.8, "max_lon": 75.2}

# India calibrated thresholds (threshold_calibration_india_mangalore.json
# -> derived_thresholds), NOT the Gulf hand values.
# v2: SAR-vs-AIS classification still uses the full AIS pool, but the
# trajectory + ST-DBSCAN behaviour stages are restricted to the merchant
# fleet -- the raw run drowned in the New Mangalore artisanal fishing fleet
# (FISHING+GEAR+net-buoy "OTHER" = ~60k of 105k AIS bins -> 1528 tier-1 flags).
run_window(
    "india_mangalore_dec2025_jan2026", BBOX,
    "2025-12-01", "2026-01-15",
    matched_km=2.6434, spoofed_km=20.2469,
    eps_km=8.9773, eps_h=6.0,
    max_pulls=8,
    skip_fetch=True,
    keep_vessel_types={"CARGO", "BUNKER", "PASSENGER", "SEISMIC_VESSEL", "OTHER"},
)
