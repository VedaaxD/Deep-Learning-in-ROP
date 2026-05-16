"""
XAI Zone Scoring — 3 Methods, Signed + Intensity Ratio
Three XAI methods, each with appropriate scoring:

┌─────────────┬──────────────────┬────────────────────────────────────────┐
│ Method      │ Map type         │ Scoring approach                       │
├─────────────┼──────────────────┼────────────────────────────────────────┤
│ GradCAM++   │ Always positive  │ Intensity Ratio (plain saliency        │
│             │ (0 to 1)         │ density — no signed split needed)      │
├─────────────┼──────────────────┼────────────────────────────────────────┤
│ GradSHAP    │ Signed           │ Signed Saliency Density                │
│             │ (negative to     │ Sd(+) = toward ROP                     │
│             │  positive)       │ Sd(-) = away from ROP                  │
├─────────────┼──────────────────┼────────────────────────────────────────┤
│ Occlusion   │ Signed           │ Same as GradSHAP                       │
│             │ (negative to     │ Sd(+) = hiding this region drops conf  │
│             │  positive)       │ Sd(-) = hiding this region raises conf │
└─────────────┴──────────────────┴────────────────────────────────────────┘

Formula for GradCAM++ (Intensity Ratio Score):
    heatmap_norm   = normalize(gradcam_map, 0, 1)
    threshold      = 90th percentile of heatmap_norm
    binary_mask    = heatmap_norm > threshold
    Sd(zone)       = |binary_mask ∩ zone_mask| / |binary_mask|

Formula for GradSHAP / Occlusion (Signed Saliency Density):
    pos_mask       = top 10% of positive values → pixels pushing TOWARD ROP
    neg_mask       = bottom 10% of negative values → pixels pushing AWAY
    Sd_pos(zone)   = |pos_mask ∩ zone_mask| / |pos_mask|
    Sd_neg(zone)   = |neg_mask ∩ zone_mask| / |neg_mask|

Zones:
    Zone I  → 2× OD radius (ICROP definition — most critical)
    Zone II → 2r to 5r annulus
    Rest    → Ridge + Periphery combined
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

zone_mask_root = "/home/veda/rop_explainability/rop_latest_with_cv/zone_outputs"
viz_root = "./Visualization_Results"
model_input_size = 256
threshold_percentile = 90  # top 10% = most activated pixels
output_dir = "./XAI_Zone_Scores"

runs= [
    (3, "V_Augmented_ResNet18_pretrained", "UHO"),
    (3, "V_Augmented_EfficientNetB0_pretrained", "UHO"),
    (4, "K_Augmented_ResNet18_pretrained", "VIIO"),
    (4, "K_Augmented_EfficientNetB0_pretrained", "VIIO"),
]

zone_keys = ["zone_i", "zone_ii", "rest"]
zone_labels = ["Zone I", "Zone II", "Rest (Ridge+Periphery)"]

def load_zone_masks(image_path: str, dataset_name: str) -> dict:
    """
    Load Zone I, Zone II, Rest (Ridge+Periphery) boolean masks.
    Resized to model_input_size. Returns None if any file is missing.
    """
    stem = Path(image_path).stem
    label_dir = "Positive" if "Positive" in image_path else "Negative"
    masks_dir = Path(zone_mask_root) / dataset_name / label_dir / "masks"

    masks = {}
    for zone in ["zone_i", "zone_ii"]:
        f = masks_dir / f"{stem}_{zone}.png"
        if not f.exists():
            return None
        raw = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        masks[zone] = cv2.resize(raw, (model_input_size, model_input_size),
                                 interpolation=cv2.INTER_NEAREST) > 0

    rest = np.zeros((model_input_size, model_input_size), dtype=bool)
    for zone in ["ridge", "periphery"]:
        f = masks_dir / f"{stem}_{zone}.png"
        if not f.exists():
            return None
        raw = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        rest = rest | (cv2.resize(raw, (model_input_size, model_input_size),
                                  interpolation=cv2.INTER_NEAREST) > 0)
    masks["rest"] = rest
    return masks


#Scoring Method 1: GradCAM++ Intensity Ratio 

def score_gradcam(gradcam_map: np.ndarray, zone_masks: dict) -> dict:
    """
    GradCAM++ scoring — Intensity Ratio Score.

    GradCAM++ values are always non-negative (0 to 1), so no signed
    split is needed. We simply threshold at the 90th percentile and
    compute what fraction of activated pixels fall in each zone.

    Steps:
        1. Ensure 2D (H, W) — GradCAM++ already outputs grayscale
        2. Normalize to [0, 1] (usually already done by library)
        3. Threshold at 90th percentile → binary activation mask
        4. Sd(zone) = overlap / total activated pixels
    """
    # Step 1: ensure 2D
    if gradcam_map.ndim == 3:
        gradcam_map = gradcam_map.mean(axis=2)

    # Step 2: normalize to [0, 1]
    vmin, vmax = gradcam_map.min(), gradcam_map.max()
    if vmax - vmin < 1e-8:
        return {f"gc_{z}": 0.0 for z in zone_keys}
    norm_map = (gradcam_map - vmin) / (vmax - vmin)

    # Step 3: threshold at 90th percentile
    threshold = np.percentile(norm_map, threshold_percentile)
    binary_mask = norm_map > threshold
    total_active = binary_mask.sum()

    if total_active == 0:
        return {f"gc_{z}": 0.0 for z in zone_keys}

    # Step 4: saliency density per zone
    scores = {}
    for zone_key, zone_mask in zone_masks.items():
        overlap = (binary_mask & zone_mask).sum()
        scores[f"gc_{zone_key}"] = float(overlap) / float(total_active)

    return scores


#Scoring Method 2 & 3: Signed Saliency Density (GradSHAP + Occlusion)

def score_signed(attr_map: np.ndarray, zone_masks: dict,
                 prefix: str) -> dict:
    """
    Signed saliency density for GradSHAP or Occlusion.

    Because these methods produce both positive and negative values:
        Positive values - pixel pushes model TOWARD ROP (green in heatmap)
        Negative values -pixel pushes model AWAY from ROP (red in heatmap)

    We compute two separate density scores per zone:
        Sd_pos(zone) = top 10% positive pixels ∩ zone / total top-10% positive
        Sd_neg(zone) = bottom 10% negative pixels ∩ zone / total bottom-10% neg

    prefix = "gs" for GradSHAP, "occ" for Occlusion
    """
    if attr_map.ndim == 3:
        attr_map = attr_map.mean(axis=2)

    scores = {}

    pos_vals = attr_map[attr_map > 0]
    if len(pos_vals) > 0:
        pos_thresh = np.percentile(pos_vals, threshold_percentile)
        pos_mask = (attr_map > 0) & (attr_map >= pos_thresh)
        total_pos = pos_mask.sum()
    else:
        pos_mask = np.zeros_like(attr_map, dtype=bool)
        total_pos = 0

    for zone_key, zone_mask in zone_masks.items():
        if total_pos > 0:
            scores[f"{prefix}_{zone_key}_pos"] = (
                    float((pos_mask & zone_mask).sum()) / float(total_pos))
        else:
            scores[f"{prefix}_{zone_key}_pos"] = 0.0

    neg_vals = attr_map[attr_map < 0]
    if len(neg_vals) > 0:
        neg_thresh = np.percentile(neg_vals, 100 - threshold_percentile)
        neg_mask = (attr_map < 0) & (attr_map <= neg_thresh)
        total_neg = neg_mask.sum()
    else:
        neg_mask = np.zeros_like(attr_map, dtype=bool)
        total_neg = 0

    for zone_key, zone_mask in zone_masks.items():
        if total_neg > 0:
            scores[f"{prefix}_{zone_key}_neg"] = (
                    float((neg_mask & zone_mask).sum()) / float(total_neg))
        else:
            scores[f"{prefix}_{zone_key}_neg"] = 0.0

    return scores

def load_npy_files(npy_dir: str, sample_idx: int):
    """Load all 3 .npy files saved by visualization.py. sample_idx is 1-based."""
    gc_path = os.path.join(npy_dir, f"sample_img_{sample_idx}_gradcam.npy")
    gs_path = os.path.join(npy_dir, f"sample_img_{sample_idx}_gradshap.npy")
    occ_path = os.path.join(npy_dir, f"sample_img_{sample_idx}_occlusion.npy")

    if not all(os.path.exists(p) for p in [gc_path, gs_path, occ_path]):
        return None, None, None

    return np.load(gc_path), np.load(gs_path), np.load(occ_path)


def compute_consistency(row):
    """
    Consistency score: do GradCAM++, GradSHAP(+), Occlusion(+) 
    agree on which zone is dominant?
    Returns:
        dominant_zone : which zone got highest avg attention
        consistency   : 1.0 if all agree, 0.5 if 2/3 agree, 0.0 if all disagree
        agreement_pct : float 0-1
    """
    zones = ["zone_i", "zone_ii", "rest"]

    #Dominant zone per method
    gc_dom = max(zones, key=lambda z: row.get(f"gc_{z}", 0))
    gs_dom = max(zones, key=lambda z: row.get(f"gs_{z}_pos", 0))
    occ_dom = max(zones, key=lambda z: row.get(f"occ_{z}_pos", 0))

    votes = [gc_dom, gs_dom, occ_dom]

    # Majority vote
    from collections import Counter
    dominant_zone = Counter(votes).most_common(1)[0][0]
    agree_count = votes.count(dominant_zone)
    consistency = agree_count / 3.0

    return dominant_zone, consistency

def score_one_run(exp_no, model_name, dataset_name) -> pd.DataFrame:
    """Score all TP/TN/FP/FN samples for one run. Returns DataFrame."""
    rows = []
    base_dir = os.path.join(viz_root, f"Exp{exp_no}", model_name)

    if not os.path.exists(base_dir):
        print(f"  [SKIP] Not found: {base_dir}")
        return pd.DataFrame()

    for category in ["TP", "TN", "FP", "FN"]:
        cat_dir = os.path.join(base_dir, category)
        npy_dir = os.path.join(cat_dir, "raw_attributions")
        paths_file = os.path.join(cat_dir, "image_paths.txt")

        if not os.path.exists(paths_file):
            print(f"  [WARN] No image_paths.txt in {cat_dir}")
            print(f"         Re-run main.py to generate raw attributions.")
            continue

        with open(paths_file) as f:
            image_paths = [l.strip() for l in f if l.strip()]

        print(f"Scoring {len(image_paths)} {category} images...")

        for idx, img_path in enumerate(image_paths, 1):

            masks = load_zone_masks(img_path, dataset_name)
            if masks is None:
                print(f"    [SKIP] Zone masks missing: {Path(img_path).name}")
                continue

            gc_map, gs_map, occ_map = load_npy_files(npy_dir, idx)
            if gc_map is None:
                print(f"    [SKIP] .npy files missing for sample {idx}")
                continue

            row = {
                "exp": exp_no,
                "model": model_name,
                "dataset": dataset_name,
                "category": category,
                "image": Path(img_path).name,
                "rop_label": 1 if category in ["TP", "FN"] else 0,
            }

            # GradCAM++ → intensity ratio (unsigned)
            row.update(score_gradcam(gc_map, masks))

            # GradSHAP → signed saliency density
            row.update(score_signed(gs_map, masks, prefix="gs"))

            # Occlusion → signed saliency density
            row.update(score_signed(occ_map, masks, prefix="occ"))

            dominant_zone, consistency = compute_consistency(row)
            row["dominant_zone"] = dominant_zone
            row["xai_consistency"] = round(consistency, 3)
            rows.append(row)

    return pd.DataFrame(rows)


def build_clinician_summary(df):
    """
    For each category (TP/TN/FP/FN), report:
      - Where did the model predominantly focus?
      - How consistently across XAI methods?
      - What % of cases had consistent focus?
    No ground truth assumed — purely descriptive.
    """
    rows = []
    for category in ["TP", "TN", "FP", "FN"]:
        sub = df[df["category"] == category]
        if sub.empty:
            continue

        total = len(sub)

        # Dominant zone distribution
        zone_counts = sub["dominant_zone"].value_counts()
        dominant = zone_counts.idxmax() if not zone_counts.empty else "unknown"
        dominant_pct = zone_counts.max() / total * 100 if not zone_counts.empty else 0

        # Consistency
        avg_consistency = sub["xai_consistency"].mean()
        high_consist = (sub["xai_consistency"] == 1.0).sum()

        rows.append({
            "Category": category,
            "N": total,
            "Dominant Focus Zone": dominant.replace("_", " ").title(),
            "% Cases w/ that focus": round(dominant_pct, 1),
            "Avg XAI Consistency": round(avg_consistency, 3),
            "Fully Consistent Cases": f"{high_consist}/{total}",
        })

    return pd.DataFrame(rows)

def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, label_name in [(1, "ROP Positive"), (0, "ROP Negative")]:
        sub = df[df["rop_label"] == label]
        if sub.empty:
            continue
        for zone_key, zone_label in zip(zone_keys, zone_labels):
            rows.append({
                "Class": label_name,
                "Region": zone_label,
                # GradCAM++ — single unsigned score
                "GradCAM++ Sd": round(sub[f"gc_{zone_key}"].mean(), 4),
                # GradSHAP — signed
                "GradSHAP Sd(+)": round(sub[f"gs_{zone_key}_pos"].mean(), 4),
                "GradSHAP Sd(-)": round(sub[f"gs_{zone_key}_neg"].mean(), 4),
                # Occlusion — signed
                "Occlusion Sd(+)": round(sub[f"occ_{zone_key}_pos"].mean(), 4),
                "Occlusion Sd(-)": round(sub[f"occ_{zone_key}_neg"].mean(), 4),
                "N images": len(sub),
            })
    return pd.DataFrame(rows)

def plot_summary(summary: pd.DataFrame, save_path: str, title: str):
    """
    2×3 panel plot:
        Rows    = ROP Positive / ROP Negative
        Columns = GradCAM++ / GradSHAP / Occlusion
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    fig.suptitle(title, fontsize=11, fontweight="bold")

    x = np.arange(len(zone_labels))
    width = 0.28
    xlabels = ["Zone I", "Zone II", "Rest"]

    for row_idx, (label, label_name) in enumerate(
            [(1, "ROP Positive"), (0, "ROP Negative")]):

        sub = summary[summary["Class"] == label_name]
        if sub.empty:
            continue

        def vals(col):
            return [sub[sub["Region"] == z][col].values[0] for z in zone_labels]

        # ── Column 0: GradCAM++ (single bar — unsigned) ──
        ax = axes[row_idx][0]
        ax.bar(x, vals("GradCAM++ Sd"), width * 2,
               color="#9b59b6", alpha=0.85, label="GradCAM++ Sd")
        ax.axhline(0.33, color="gray", linestyle="--", linewidth=0.8,
                   alpha=0.6, label="Random baseline")
        ax.set_title(f"GradCAM++ — {label_name}", fontsize=9)
        ax.set_xticks(x);
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_ylabel("Saliency Density Sd");
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7);
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        # ── Column 1: GradSHAP (signed bars) ──
        ax = axes[row_idx][1]
        ax.bar(x - width / 2, vals("GradSHAP Sd(+)"), width,
               color="#27ae60", alpha=0.85, label="Sd(+) toward ROP")
        ax.bar(x + width / 2, vals("GradSHAP Sd(-)"), width,
               color="#e74c3c", alpha=0.85, label="Sd(-) away from ROP")
        ax.axhline(0.33, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_title(f"GradSHAP — {label_name}", fontsize=9)
        ax.set_xticks(x);
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_ylim(0, 1);
        ax.legend(fontsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        #Column 2: Occlusion (signed bars) 
        ax = axes[row_idx][2]
        ax.bar(x - width / 2, vals("Occlusion Sd(+)"), width,
               color="#27ae60", alpha=0.85, label="Sd(+) toward ROP")
        ax.bar(x + width / 2, vals("Occlusion Sd(-)"), width,
               color="#e74c3c", alpha=0.85, label="Sd(-) away from ROP")
        ax.axhline(0.33, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_title(f"Occlusion — {label_name}", fontsize=9)
        ax.set_xticks(x);
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_ylim(0, 1);
        ax.legend(fontsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot -> {save_path}")

def main():
    os.makedirs(output_dir, exist_ok=True)
    all_dfs = []

    for exp_no, model_name, dataset_name in RUNS:
        print(f"\n{'=' * 60}")
        print(f"Exp{exp_no} | {model_name} | test={dataset_name}")
        print(f"{'=' * 60}")

        df = score_one_run(exp_no, model_name, dataset_name)
        if df.empty:
            continue

        all_dfs.append(df)
        df.to_csv(os.path.join(output_dir,
                               f"Exp{exp_no}_{model_name}_per_image.csv"),
                  index=False)

        summary = build_summary_table(df)
        summary.to_csv(os.path.join(output_dir,
                                    f"Exp{exp_no}_{model_name}_summary.csv"),
                       index=False)

        print("\n── Summary ──")
        print(summary.to_string(index=False, float_format="{:.4f}".format))

        plot_summary(
            summary,
            save_path=os.path.join(output_dir,
                                   f"Exp{exp_no}_{model_name}_summary.png"),
            title=(f"Exp{exp_no} | {model_name} | Test: {dataset_name}\n"
                   f"Saliency Density @ {100 - threshold_percentile}% threshold")
        )

    if all_dfs:
        pd.concat(all_dfs, ignore_index=True).to_csv(
            os.path.join(output_dir, "all_runs_combined.csv"), index=False)
        print(f"\nAll results → {output_dir}/all_runs_combined.csv")


if __name__ == "__main__":
    main()
