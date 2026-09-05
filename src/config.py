import os

from dotenv import load_dotenv

load_dotenv()

GFW_API_TOKEN = os.environ.get("GFW_API_TOKEN")
AISSTREAM_API_KEY = os.environ.get("AISSTREAM_API_KEY")

GFW_API_BASE_URL = "https://gateway.api.globalfishingwatch.org/v3"


def require_gfw_token():
    if not GFW_API_TOKEN:
        raise RuntimeError(
            "GFW_API_TOKEN is not set. Copy .env.example to .env and add your token "
            "from https://globalfishingwatch.org/our-apis/tokens"
        )
    return GFW_API_TOKEN


# --- Test run parameters ---------------------------------------------------
# Qatar GNSS spoofing incident window (documented). Region covers Gulf waters
# off Qatar where widespread AIS/GNSS spoofing was reported.
RUN_LABEL = "qatar_gnss_spoofing_oct2025"

TEST_BBOX = {
    "min_lat": 24.5,
    "max_lat": 26.5,
    "min_lon": 51.0,
    "max_lon": 52.5,
}

TEST_START_DATE = "2025-10-03"
TEST_END_DATE = "2025-10-07"

# --- Data source caveats (verified against live API on 2026-09-03) --------
# GFW's public API does NOT expose raw AIS pings or a per-vessel track
# endpoint (confirmed: /vessels/{id}/tracks, /vessels/{id}/track, and
# /4wings/track all 404). The only AIS positional data available is the
# 4wings "public-global-presence:latest" report: one row per vessel per
# ~1/100-degree grid cell per HOURLY bucket, with entryTimestamp/
# exitTimestamp spanning the vessel's dwell time in that cell (can be many
# hours). SAR detections (public-global-sar-presence:latest) are much finer
# individual events with precise entryTimestamp/exitTimestamp.
#
# Consequence: "AIS interpolation" in this pipeline interpolates between
# hourly grid-cell presence bins (cell centroid + bin time), NOT between raw
# pings. This first version validates matched/unmatched/discrepant
# CLASSIFICATION, not precise spoofing-distance quantification -- a finer
# AIS source would be needed for the latter.
GFW_SPATIAL_RESOLUTION = "HIGH"  # ~1/100 degree (~1.1 km) grid cells
GFW_TEMPORAL_RESOLUTION = "HOURLY"
SAR_DATASET = "public-global-sar-presence:latest"
AIS_PRESENCE_DATASET = "public-global-presence:latest"
