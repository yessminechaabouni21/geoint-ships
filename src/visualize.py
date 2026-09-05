"""Step 8: interactive folium map of SAR detections and AIS tracks, color-coded by classification."""
import folium
import pandas as pd

COLORS = {
    "matched": "green",
    "discrepant": "red",
    "unmatched": "orange",
}


def build_map(match_df, ais_df, center=None, zoom_start=8):
    if center is None:
        center = [match_df["sar_lat"].mean(), match_df["sar_lon"].mean()]

    m = folium.Map(location=center, zoom_start=zoom_start, tiles="OpenStreetMap")

    for classification, color in COLORS.items():
        layer = folium.FeatureGroup(name=f"SAR: {classification}")
        subset = match_df[match_df["classification"] == classification]
        for _, row in subset.iterrows():
            popup = (
                f"SAR id: {row['sar_id']}<br>"
                f"Timestamp: {row['sar_timestamp']}<br>"
                f"MMSI: {row['mmsi']}<br>"
                f"Ship: {row.get('ship_name', '')}<br>"
                f"Classification: {row['classification']}<br>"
                f"Distance to AIS: {row['distance_km']:.2f} km" if pd.notna(row["distance_km"]) else
                f"SAR id: {row['sar_id']}<br>Classification: {row['classification']}"
            )
            folium.CircleMarker(
                location=[row["sar_lat"], row["sar_lon"]],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                popup=folium.Popup(popup, max_width=300),
            ).add_to(layer)

            if pd.notna(row["ais_lat"]) and pd.notna(row["ais_lon"]):
                folium.PolyLine(
                    locations=[[row["sar_lat"], row["sar_lon"]], [row["ais_lat"], row["ais_lon"]]],
                    color=color,
                    weight=1,
                    dash_array="4",
                    opacity=0.6,
                ).add_to(layer)
                folium.CircleMarker(
                    location=[row["ais_lat"], row["ais_lon"]],
                    radius=3,
                    color="blue",
                    fill=True,
                    fill_opacity=0.6,
                    popup=f"AIS-interpolated position for MMSI {row['mmsi']}",
                ).add_to(layer)

        layer.add_to(m)

    ais_layer = folium.FeatureGroup(name="AIS presence tracks", show=False)
    for mmsi, track in ais_df.groupby("mmsi"):
        track = track.sort_values("timestamp")
        if len(track) < 2:
            continue
        folium.PolyLine(
            locations=track[["lat", "lon"]].values.tolist(),
            color="steelblue",
            weight=1,
            opacity=0.4,
        ).add_to(ais_layer)
    ais_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


if __name__ == "__main__":
    from src.config import TEST_BBOX, TEST_START_DATE, TEST_END_DATE, RUN_LABEL
    from src.fetch_sar import fetch_sar_detections
    from src.fetch_ais import fetch_ais_positions
    from src.match import match_and_classify

    sar_df = fetch_sar_detections(TEST_BBOX, TEST_START_DATE, TEST_END_DATE)
    ais_df = fetch_ais_positions(TEST_BBOX, TEST_START_DATE, TEST_END_DATE)
    match_df = match_and_classify(sar_df, ais_df)

    m = build_map(match_df, ais_df)
    out_path = f"data/processed/{RUN_LABEL}_map.html"
    m.save(out_path)
    print(f"Saved map to {out_path}")
