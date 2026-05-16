"""
Clinical XAI Visualization
For each sample image, produces one clean figure:

    Panel 1: Original image + Zone I/II circles drawn
    Panel 2: GradCAM++ heatmap overlaid on image + zone circles
    Panel 3: GradSHAP heatmap (signed, red=against, green=for ROP)
    Panel 4: Occlusion heatmap (signed)
    Right side: Score table (Saliency Density per zone per method)

A clinician should look at this and say:
    "The model is focusing on Zone I — that is the right place."
Usage:
    python xai_visualize_clinical.py
Output:
    ./XAI_Clinical_Figures/
        Exp3_V_Augmented_ResNet18_pretrained_TP/
            sample_1_clinical.png
            sample_2_clinical.png
            ...
        ...
"""
import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path

zone_mask_root = "/home/veda/rop_explainability/rop_latest_with_cv/zone_outputs"
viz_root        = "./Visualization_Results"
od_params_root = "/home/veda/rop_explainability/rop_latest_with_cv/zone_outputs"
output_dir       = "./XAI_Clinical_Figures"
model_input_size = 256
threshold_percentile = 90

# runs to visualize
runs = [
    (3, "V_Augmented_ResNet18_pretrained",        "UHO"),
    (3, "V_Augmented_EfficientNetB0_pretrained",  "UHO"),
    (4, "K_Augmented_ResNet18_pretrained",        "VIIO"),
    (4, "K_Augmented_EfficientNetB0_pretrained",  "VIIO"),
]

# categories to visualize (you can add "TN","FP","FN")
categories = ["TP", "TN", "FP", "FN"]

# Zone circle colors for drawing on image
zone_circle_colors= {
    "zone_i":    (255,  50,  50),   # red
    "zone_ii":   ( 50, 180, 255),   # blue
}

def load_od_params(dataset_name: str, label_dir: str) -> dict:
    """Load OD center + radius from od_params.json."""
    json_path = Path(od_params_root) / dataset_name / label_dir / "od_params.json"
    if not json_path.exists():
        return {}
    with open(json_path) as f:
        return json.load(f)
def load_original_image_with_zones(image_path: str,
                                    dataset_name: str) -> np.ndarray:
    """
    Load original image (BGR) and draw Zone I + Zone II circles on it.
    Returns RGB numpy array.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    label_dir  = "Positive" if "Positive" in image_path else "Negative"
    od_params  = load_od_params(dataset_name, label_dir)
    fname      = Path(image_path).name

    if fname in od_params:
        od   = od_params[fname]
        cx   = od["cx"]        # ← THIS LINE WAS MISSING
        cy   = od["cy"]        # ← THIS LINE WAS MISSING  
        r    = od["radius"] 
        cx, cy, r =int(round(cx)), int(round(cy)), int(round(r))

        # Draw OD disc itself (white)
        cv2.circle(img, (cx, cy), r,         (255, 255, 255), 1)
        cv2.circle(img, (cx, cy), 2 * r,     zone_circle_colors["zone_i"],  2)
        cv2.circle(img, (cx, cy), 5 * r,     zone_circle_colors["zone_ii"], 2)
        cv2.circle(img, (cx, cy), 3,         (255, 255, 255), -1)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def resize_attr_to_original(attr_map: np.ndarray,
                              target_shape: tuple) -> np.ndarray:
    """
    Resize attribution map from model_input_size back to original image size.
    target_shape = (H, W)
    """
    if attr_map.ndim == 3:
        attr_map = attr_map.mean(axis=2)   # collapse channels

    H, W = target_shape
    return cv2.resize(attr_map.astype(np.float32), (W, H),
                      interpolation=cv2.INTER_LINEAR)


def make_binary_mask(attr_map: np.ndarray, signed: bool = False,
                     side: str = "pos") -> np.ndarray:
    """
    Threshold attribution map at 90th percentile.
    signed=True → separate positive and negative thresholding
    Returns float array 0/1.
    """
    if not signed:
        vmin, vmax = attr_map.min(), attr_map.max()
        if vmax - vmin < 1e-8:
            return np.zeros_like(attr_map)
        norm = (attr_map - vmin) / (vmax - vmin)
        thresh = np.percentile(norm, threshold_percentile)
        return (norm > thresh).astype(float)
    else:
        if side == "pos":
            pos_vals = attr_map[attr_map > 0]
            if len(pos_vals) == 0:
                return np.zeros_like(attr_map)
            thresh = np.percentile(pos_vals, threshold_percentile)
            return ((attr_map > 0) & (attr_map >= thresh)).astype(float)
        else:
            neg_vals = attr_map[attr_map < 0]
            if len(neg_vals) == 0:
                return np.zeros_like(attr_map)
            thresh = np.percentile(neg_vals, 100 - threshold_percentile)
            return ((attr_map < 0) & (attr_map <= thresh)).astype(float)

def compute_scores_for_display(gc_map, gs_map, occ_map,
                                 image_path, dataset_name) -> dict:
    """
    Compute zone scores for the score table displayed in the figure.
    Returns nested dict: scores[method][zone] = value
    """
    stem      = Path(image_path).stem
    label_dir = "Positive" if "Positive" in image_path else "Negative"
    masks_dir = Path(zone_mask_root) / dataset_name / label_dir / "masks"

    # Load and resize zone masks to model input size
    zone_masks = {}
    for zone in ["zone_i", "zone_ii", "rest"]:
        if zone == "rest":
            rest = np.zeros((model_input_size, model_input_size), dtype=bool)
            for z in ["ridge", "periphery"]:
                f = masks_dir / f"{stem}_{z}.png"
                if f.exists():
                    raw  = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                    rest = rest | (cv2.resize(raw,
                                              (model_input_size, model_input_size),
                                              interpolation=cv2.INTER_NEAREST) > 0)
            zone_masks["rest"] = rest
        else:
            f = masks_dir / f"{stem}_{zone}.png"
            if not f.exists():
                return None
            raw             = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            zone_masks[zone] = cv2.resize(raw,
                                           (model_input_size, model_input_size),
                                           interpolation=cv2.INTER_NEAREST) > 0

    def sd(attr, mask, signed=False, side="pos"):
        """Saliency density = overlap / total activated."""
        a2d = attr.mean(axis=2) if attr.ndim == 3 else attr
        bm  = make_binary_mask(a2d, signed=signed, side=side)
        total = bm.sum()
        if total == 0:
            return 0.0
        return float((bm.astype(bool) & mask).sum()) / float(total)

    scores = {}
    for zone_key in ["zone_i", "zone_ii", "rest"]:
        m = zone_masks[zone_key]
        scores[zone_key] = {
            "GradCAM++":     sd(gc_map,  m, signed=False),
            "GradSHAP (+)":  sd(gs_map,  m, signed=True,  side="pos"),
            "GradSHAP (-)":  sd(gs_map,  m, signed=True,  side="neg"),
            "Occlusion (+)": sd(occ_map, m, signed=True,  side="pos"),
            "Occlusion (-)": sd(occ_map, m, signed=True,  side="neg"),
        }
    return scores

def draw_clinical_figure(original_with_zones: np.ndarray,
                         gc_map: np.ndarray,
                         gs_map: np.ndarray,
                         occ_map: np.ndarray,
                         scores: dict,
                         image_name: str,
                         category: str,
                         save_path: str):
    """
    Draw the full clinical figure:
        Col 1: Original + zones
        Col 2: GradCAM++ overlay + zones
        Col 3: GradSHAP heatmap
        Col 4: Occlusion heatmap
        Col 5: Score table
    """
    H, W = original_with_zones.shape[:2]
    # Resize attribution maps to original image size
    gc_resized  = resize_attr_to_original(gc_map, (H, W))
    gs_resized  = resize_attr_to_original(gs_map, (H, W))
    occ_resized = resize_attr_to_original(occ_map, (H, W))

    fig = plt.figure(figsize=(22, 6), dpi=150)
    fig.patch.set_facecolor("#1a1a2e")

    gs_layout = gridspec.GridSpec(
        1, 5, figure=fig,
        width_ratios=[1, 1, 1, 1, 1.1],
        wspace=0.05
    )

    label_text = "ROP Positive" if category in ["TP", "FN"] else "ROP Negative"

    fig.suptitle(
        f"{image_name}  |  {label_text} ({category})",
        fontsize=10,
        color="white",
        fontweight="bold",
        y=1.01
    )

    #Panel 1: Original img
    ax1 = fig.add_subplot(gs_layout[0, 0])
    ax1.imshow(original_with_zones)
    ax1.set_title("Original + Zones", color="white", fontsize=8, pad=4)
    ax1.axis("off")

    legend_patches = [
        mpatches.Patch(color=(1, 0.2, 0.2), label="Zone I boundary"),
        mpatches.Patch(color=(0.2, 0.7, 1), label="Zone II boundary"),
    ]
    ax1.legend(
        handles=legend_patches,
        loc="lower left",
        fontsize=6,
        framealpha=0.6,
        facecolor="#1a1a2e",
        labelcolor="white"
    )

    #Panel 2: GradCAM++ 
    ax2 = fig.add_subplot(gs_layout[0, 1])
    ax2.imshow(original_with_zones)

    gc_norm = (gc_resized - gc_resized.min()) / (
        gc_resized.max() - gc_resized.min() + 1e-8
    )

    ax2.imshow(gc_norm, cmap="jet", alpha=0.55, vmin=0, vmax=1)
    ax2.set_title("GradCAM++", color="#a78bfa", fontsize=8, pad=4)
    ax2.axis("off")

    plt.colorbar(
        ScalarMappable(norm=Normalize(0, 1), cmap="jet"),
        ax=ax2,
        fraction=0.03,
        pad=0.02
    ).ax.tick_params(labelcolor="white", labelsize=6)

    #Panel 3: GradSHAP
    ax3 = fig.add_subplot(gs_layout[0, 2])
    ax3.imshow(original_with_zones, alpha=0.35)

    gs_sym = max(abs(gs_resized.min()), abs(gs_resized.max())) + 1e-8

    ax3.imshow(
        gs_resized,
        cmap="RdYlGn",
        alpha=0.75,
        vmin=-gs_sym,
        vmax=gs_sym
    )

    ax3.set_title(
        "GradSHAP (green=ROP↑ red=ROP↓)",
        color="#6ee7b7",
        fontsize=7,
        pad=4
    )
    ax3.axis("off")

    plt.colorbar(
        ScalarMappable(norm=Normalize(-gs_sym, gs_sym), cmap="RdYlGn"),
        ax=ax3,
        fraction=0.03,
        pad=0.02
    ).ax.tick_params(labelcolor="white", labelsize=6)

    #Panel 4: Occlusion
    ax4 = fig.add_subplot(gs_layout[0, 3])
    ax4.imshow(original_with_zones, alpha=0.35)

    occ_sym = max(abs(occ_resized.min()), abs(occ_resized.max())) + 1e-8

    ax4.imshow(
        occ_resized,
        cmap="RdYlGn",
        alpha=0.75,
        vmin=-occ_sym,
        vmax=occ_sym
    )

    ax4.set_title(
        "Occlusion (green=ROP↑ red=ROP↓)",
        color="#6ee7b7",
        fontsize=7,
        pad=4
    )
    ax4.axis("off")

    plt.colorbar(
        ScalarMappable(norm=Normalize(-occ_sym, occ_sym), cmap="RdYlGn"),
        ax=ax4,
        fraction=0.03,
        pad=0.02
    ).ax.tick_params(labelcolor="white", labelsize=6)

    #Panel 5: Score Table
    ax5 = fig.add_subplot(gs_layout[0, 4])
    ax5.set_facecolor("#0f0f23")
    ax5.axis("off")

    if scores:
        zone_display = {
            "zone_i": "Zone I",
            "zone_ii": "Zone II",
            "rest": "Rest"
        }

        methods = [
            "GradCAM++",
            "GradSHAP (+)",
            "GradSHAP (-)",
            "Occlusion (+)",
            "Occlusion (-)"
        ]

        method_colors = {
            "GradCAM++": "#a78bfa",
            "GradSHAP (+)": "#6ee7b7",
            "GradSHAP (-)": "#f87171",
            "Occlusion (+)": "#6ee7b7",
            "Occlusion (-)": "#f87171",
        }

        y_start = 0.97

        ax5.text(
            0.5, y_start,
            "Saliency Density Scores",
            transform=ax5.transAxes,
            fontsize=8,
            color="white",
            ha="center",
            fontweight="bold"
        )

        ax5.text(
            0.5, y_start - 0.06,
            f"(top {100-threshold_percentile}% activated pixels)",
            transform=ax5.transAxes,
            fontsize=6,
            color="#888",
            ha="center"
        )

        y = y_start - 0.14

        ax5.text(
            0.02, y, "Method",
            transform=ax5.transAxes,
            fontsize=6.5,
            color="#aaa",
            fontweight="bold"
        )

        for col_idx, zone_key in enumerate(["zone_i", "zone_ii", "rest"]):
            x_pos = 0.38 + col_idx * 0.21
            ax5.text(
                x_pos, y, zone_display[zone_key],
                transform=ax5.transAxes,
                fontsize=6.5,
                color="#aaa",
                fontweight="bold",
                ha="center"
            )

        y -= 0.06
        ax5.axhline(y=y + 0.02, xmin=0.02, xmax=0.98, color="#333", linewidth=0.8)

        for method in methods:
            y -= 0.09

            ax5.text(
                0.02, y, method,
                transform=ax5.transAxes,
                fontsize=6.5,
                color=method_colors[method]
            )

            for col_idx, zone_key in enumerate(["zone_i", "zone_ii", "rest"]):
                x_pos = 0.38 + col_idx * 0.21
                val = scores[zone_key][method]

                intensity = min(val * 2, 1.0)

                if "(-)" in method:
                    cell_color = (
                        1.0,
                        1.0 - intensity * 0.6,
                        1.0 - intensity * 0.6
                    )
                else:
                    cell_color = (
                        1.0 - intensity * 0.6,
                        1.0,
                        1.0 - intensity * 0.6
                    )

                ax5.text(
                    x_pos, y, f"{val:.3f}",
                    transform=ax5.transAxes,
                    fontsize=7,
                    color=cell_color,
                    ha="center",
                    fontweight="bold"
                )

        # Clinical Interpretation
        y -= 0.14

        ax5.axhline(
            y=y + 0.06,
            xmin=0.02,
            xmax=0.98,
            color="#333",
            linewidth=0.8
        )

        interp, interp_col, dominant_pos,dominant_neg, consistency = get_interpretation(
            scores, category
        )

        ax5.text(
            0.5,
            y,
            interp,
            transform=ax5.transAxes,
            fontsize=7,
            color=interp_col,
            ha="center",
            fontweight="bold"
        )

    plt.tight_layout(pad=0.5)

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )

    plt.close()
def get_interpretation(scores, category):
    zones = ["zone_i", "zone_ii", "rest"]

    # Always computed the same way regardless of category
    pos_avg = {z: np.mean([
        scores[z]["GradCAM++"],
        scores[z]["GradSHAP (+)"],
        scores[z]["Occlusion (+)"],
    ]) for z in zones}

    neg_avg = {z: np.mean([
        scores[z]["GradSHAP (-)"],
        scores[z]["Occlusion (-)"],
    ]) for z in zones}

    # Dominant activation zone — consistent across all categories
    dominant_pos     = max(zones, key=lambda z: pos_avg[z])
    dominant_pos_val = pos_avg[dominant_pos]

    # Dominant suppression zone — consistent across all categories
    dominant_neg     = max(zones, key=lambda z: neg_avg[z])
    dominant_neg_val = neg_avg[dominant_neg]

    zone_labels = {
        "zone_i":  "Zone I (posterior)",
        "zone_ii": "Zone II (mid-periphery)",
        "rest":    "Periphery/Ridge",
    }
    # Consistency — always based on positive scores
    per_method = {
        "GradCAM++":     max(zones, key=lambda z: scores[z]["GradCAM++"]),
        "GradSHAP (+)":  max(zones, key=lambda z: scores[z]["GradSHAP (+)"]),
        "Occlusion (+)": max(zones, key=lambda z: scores[z]["Occlusion (+)"]),
    }
    agree_count = sum(1 for v in per_method.values() if v == dominant_pos)
    consistency = agree_count / 3
    consist_str = "consistent" if consistency == 1.0 else "inconsistent"

    #Category only changes the LABEL, not the calculation 
    category_flags = {
        "TP": "Evidence FOR ROP",
        "TN": "Evidence AGAINST ROP",
        "FP": "Spurious activation",
        "FN": "Missed ROP signal",
    }
    flag = category_flags.get(category, "")

    interp = (
        f"[{flag}]  "
        f"Activation: {zone_labels[dominant_pos]} ({dominant_pos_val:.2f})  |  "
        f"Suppression: {zone_labels[dominant_neg]} ({dominant_neg_val:.2f})  |  "
        f"XAI: {consist_str}"
    )

    interp_col = "#6ee7b7" if consistency == 1.0 else "#fbbf24"

    return interp, interp_col, dominant_pos, dominant_neg, consistency
def process_one_run(exp_no, model_name, dataset_name):
    """Generate clinical figures for all samples in one run."""
    base_dir = os.path.join(viz_root, f"Exp{exp_no}", model_name)

    if not os.path.exists(base_dir):
        print(f"  [SKIP] Not found: {base_dir}")
        return

    for category in categories:
        cat_dir    = os.path.join(base_dir, category)
        npy_dir    = os.path.join(cat_dir, "raw_attributions")
        paths_file = os.path.join(cat_dir, "image_paths.txt")

        if not os.path.exists(paths_file):
            continue

        with open(paths_file) as f:
            image_paths = [l.strip() for l in f if l.strip()]

        out_dir = os.path.join(output_dir,
                               f"Exp{exp_no}_{model_name}_{category}")
        os.makedirs(out_dir, exist_ok=True)

        print(f"  Visualizing {len(image_paths)} {category} images...")

        for idx, img_path in enumerate(image_paths, 1):
            # Load .npy attribution files
            gc_path  = os.path.join(npy_dir, f"sample_img_{idx}_gradcam.npy")
            gs_path  = os.path.join(npy_dir, f"sample_img_{idx}_gradshap.npy")
            occ_path = os.path.join(npy_dir, f"sample_img_{idx}_occlusion.npy")

            if not all(os.path.exists(p) for p in [gc_path, gs_path, occ_path]):
                print(f"    [SKIP] .npy missing for sample {idx}")
                continue

            gc_map  = np.load(gc_path)
            gs_map  = np.load(gs_path)
            occ_map = np.load(occ_path)

            # Load original image with zone circles drawn on it
            orig_with_zones = load_original_image_with_zones(img_path, dataset_name)
            if orig_with_zones is None:
                print(f"    [SKIP] Cannot load: {Path(img_path).name}")
                continue

            # Compute zone scores for the score table
            scores = compute_scores_for_display(
                gc_map, gs_map, occ_map, img_path, dataset_name)

            # Draw and save clinical figure
            save_path = os.path.join(out_dir, f"sample_{idx}_clinical.png")
            draw_clinical_figure(
                original_with_zones=orig_with_zones,
                gc_map=gc_map,
                gs_map=gs_map,
                occ_map=occ_map,
                scores=scores,
                image_name=Path(img_path).name,
                category=category,
                save_path=save_path,
            )
            print(f"    Saved → sample_{idx}_clinical.png")


def main():
    os.makedirs(output_dir, exist_ok=True)

    for exp_no, model_name, dataset_name in runs:
        print(f"\n{'='*60}")
        print(f"Exp{exp_no} | {model_name} | test={dataset_name}")
        print(f"{'='*60}")
        process_one_run(exp_no, model_name, dataset_name)

    print(f"\n✓ All figures → {output_dir}/")


if __name__ == "__main__":
    main()
