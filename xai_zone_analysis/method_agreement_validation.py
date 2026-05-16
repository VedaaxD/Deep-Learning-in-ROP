"""
Method Agreement Validation across ALL images.
Run on XAI_Zone_Scores CSV files.
"""

import pandas as pd
import numpy as np
from pathlib import Path

csv_dir = "/home/veda/rop_explainability/rop_latest_with_cv/XAI_Zone_Scores"
zones  = ["zone_i", "zone_ii", "rest"]

def dominant_zone(row, prefix):
    """Find which zone has highest score for a given method prefix."""
    if prefix == "gc":
        scores = {z: row.get(f"gc_{z}", 0) for z in zones}
    elif prefix == "gs_pos":
        scores = {z: row.get(f"gs_{z}_pos", 0) for z in zones}
    elif prefix == "occ_pos":
        scores = {z: row.get(f"occ_{z}_pos", 0) for z in zones}
    return max(scores, key=scores.get)


def compute_agreement(df):
    results = []
    for _, row in df.iterrows():
        gc  = dominant_zone(row, "gc")
        gs  = dominant_zone(row, "gs_pos")
        occ = dominant_zone(row, "occ_pos")

        results.append({
            "image":    row.get("image", ""),
            "category": row.get("category", ""),
            "gc_dom":   gc,
            "gs_dom":   gs,
            "occ_dom":  occ,
            "gc_gs":    gc == gs,
            "gc_occ":   gc == occ,
            "gs_occ":   gs == occ,
            "all3":     gc == gs == occ,
        })
    return pd.DataFrame(results)


def main():
    all_dfs = []
    for csv_file in Path(csv_dir).glob("*_per_image.csv"):
        df = pd.read_csv(csv_file)
        all_dfs.append(df)

    if not all_dfs:
        print("No CSV files found. Check csv_dir path.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    agree_df = compute_agreement(combined)

    total = len(agree_df)
    print(f"\nTotal images analysed: {total}")
    print("="*55)

    # Overall agreement
    print("\n Overall Method Agreement")
    print(f"GradCAM++ ↔ GradSHAP   : "
          f"{agree_df['gc_gs'].sum()}/{total} = "
          f"{agree_df['gc_gs'].mean()*100:.1f}%")
    print(f"GradCAM++ ↔ Occlusion  : "
          f"{agree_df['gc_occ'].sum()}/{total} = "
          f"{agree_df['gc_occ'].mean()*100:.1f}%")
    print(f"GradSHAP  ↔ Occlusion  : "
          f"{agree_df['gs_occ'].sum()}/{total} = "
          f"{agree_df['gs_occ'].mean()*100:.1f}%")
    print(f"All 3 agree            : "
          f"{agree_df['all3'].sum()}/{total} = "
          f"{agree_df['all3'].mean()*100:.1f}%")

    # Per category breakdown
    print("\nAgreement by Category")
    for cat in ["TP", "TN", "FP", "FN"]:
        sub = agree_df[agree_df["category"] == cat]
        if sub.empty:
            continue
        n = len(sub)
        print(f"\n  {cat} ({n} images):")
        print(f"    GC ↔ GS  : {sub['gc_gs'].mean()*100:.1f}%")
        print(f"    GC ↔ OCC : {sub['gc_occ'].mean()*100:.1f}%")
        print(f"    GS ↔ OCC : {sub['gs_occ'].mean()*100:.1f}%")
        print(f"    All 3    : {sub['all3'].mean()*100:.1f}%")

    # Dominant zone distribution per method
    print("\n Dominant Zone Distribution")
    for method, col in [("GradCAM++","gc_dom"),
                        ("GradSHAP","gs_dom"),
                        ("Occlusion","occ_dom")]:
        counts = agree_df[col].value_counts()
        print(f"\n  {method}:")
        for zone, count in counts.items():
            print(f"    {zone:10s}: {count}/{total} = {count/total*100:.1f}%")

    # Save
    agree_df.to_csv(f"{csv_dir}/method_agreement.csv", index=False)
    print(f"\n✓ Saved → {csv_dir}/method_agreement.csv")
if __name__ == "__main__":
    main()
