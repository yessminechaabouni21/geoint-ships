"""Re-run the fixed (uncapped, distance-tiered) classification from src.match
against the already-pulled Hormuz crisis/control raw SAR+AIS data. No
re-fetch -- reuses data/raw/hormuz_*_{sar_detections,ais_positions}.csv.
"""
import pandas as pd

from src.fetch_sar import fetch_sar_detections  # noqa: F401 (documents source shape)
from src.match import match_and_classify

RUNS = {
    "CRISIS (Mar 19-24, 2026)": "hormuz_crisis_mar2026",
    "CONTROL (Jan 17-21, 2026)": "hormuz_control_mar2026",
}

CATEGORY_ORDER = ["matched", "discrepant", "likely_spoofed", "no_ais_activity"]

if __name__ == "__main__":
    summaries = {}

    for label, run_label in RUNS.items():
        sar_df = pd.read_csv(f"data/raw/{run_label}_sar_detections.csv", parse_dates=["timestamp"])
        ais_df = pd.read_csv(f"data/raw/{run_label}_ais_positions.csv", parse_dates=["timestamp"])

        result = match_and_classify(sar_df, ais_df)
        out_path = f"data/processed/{run_label}_reclassified.csv"
        result.to_csv(out_path, index=False)

        vc = result["classification"].value_counts()
        n = len(result)
        summaries[label] = (n, vc)

        print(f"=== {label} === total SAR detections: {n}")
        for cat in CATEGORY_ORDER:
            c = vc.get(cat, 0)
            print(f"  {cat}: {c} ({100 * c / n:.1f}%)")
        print(f"  Saved to {out_path}")
        print()

    print("=== SIDE-BY-SIDE ===")
    header = f"{'category':<18}" + "".join(f"{label:>28}" for label in summaries)
    print(header)
    for cat in CATEGORY_ORDER:
        row = f"{cat:<18}"
        for label, (n, vc) in summaries.items():
            c = vc.get(cat, 0)
            row += f"{c:>10} ({100 * c / n:5.1f}%)".rjust(28)
        print(row)
