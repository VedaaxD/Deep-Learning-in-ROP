"""
 Generate ICROP-compliant zone masks from od_params.json.

ICROP Zones:
  Zone I    : radius = 2 × OD_radius  (most posterior, highest ROP risk)
  Zone II   : 2r to 5r  (intermediate)
  Ridge     : 5r to 7r  (transition)
  Periphery : > 7r     (outermost)

  rest_mask = ridge ∪ periphery  (matches xai_zone_scoring.py expectation)

Output masks: {zone_output_root}/{dataset}/{split}/masks/{stem}_{zone}.png
  zones: zone_i, zone_ii, ridge, periphery
"""

import os, json
import numpy as np
import cv2
from pathlib import Path

zone_output_root = "/home/veda/rop_explainability/rop_latest_with_cv/zone_outputs"

# ICROP radius multipliers (multiples of OD disc radius)
ZONE_I_R    = 2.0
ZONE_II_R   = 5.0
RIDGE_R     = 7.0
# Periphery: > RIDGE_R to image edge

datasets = ["UHO", "VIIO"]
splits  = ["Positive", "Negative"]

# Where to look for original images (to read actual H, W)
data_root = "/home/veda/rop_explainability/rop_latest_with_cv"

def get_img_hw(dataset, split, image_name):
    """Find the actual image and return its (H, W)."""
    stem = Path(image_name).stem
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG", ".JPEG"):
        p = Path(data_root) / dataset / split / (stem + ext)
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                return img.shape[:2]
    # fallback: try zone_output parent
    for ext in (".jpg", ".jpeg", ".png"):
        p = Path(zone_output_root).parent / "images" / (stem + ext)
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                return img.shape[:2]
    print(f"      [WARN] Image not found for sizing, defaulting to 512×512: {image_name}")
    return (512, 512)

def get_retinal_mask(H, W):
    """
    Create a circular mask of the actual retinal image area.
    The retinal image is always a circle inscribed in the frame.
    """
    cx, cy   = W // 2, H // 2
    radius   = min(H, W) // 2
    mask     = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    return mask > 0


def generate_zones(cx, cy, radius, H, W):
    """
    Generate 4 zone masks clipped to the actual retinal boundary.
    Black background pixels are excluded from ALL zones.
    """
    # Distance from every pixel to OD center
    ys, xs = np.ogrid[:H, :W]
    dist   = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)

    r1 = radius * ZONE_I_R
    r2 = radius * ZONE_II_R
    r3 = radius * RIDGE_R

    # Retinal boundary mask — excludes black background
    retinal_mask = get_retinal_mask(H, W)

    # Apply retinal mask to every zone
    return {
        "zone_i":    (retinal_mask & (dist <= r1)              ).astype(np.uint8) * 255,
        "zone_ii":   (retinal_mask & (dist > r1) & (dist <= r2)).astype(np.uint8) * 255,
        "ridge":     (retinal_mask & (dist > r2) & (dist <= r3)).astype(np.uint8) * 255,
        "periphery": (retinal_mask & (dist > r3)               ).astype(np.uint8) * 255,
    }


def process_split(dataset, split):
    od_json = Path(zone_output_root) / dataset / split / "od_params.json"
    if not od_json.exists():
        print(f"[SKIP] od_params.json not found: {od_json}")
        return

    with open(od_json) as f:
        od_params = json.load(f)

    masks_dir = Path(zone_output_root) / dataset / split / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    for image_name, params in od_params.items():
        cx = params["cx"]; cy = params["cy"]; r = params["radius"]

        if r < 1:
            print(f"[SKIP] zero radius: {image_name}")
            continue

        H, W  = get_img_hw(dataset, split, image_name)
        zones = generate_zones(cx, cy, r, H, W)
        stem  = Path(image_name).stem

        for zone_name, mask in zones.items():
            cv2.imwrite(str(masks_dir / f"{stem}_{zone_name}.png"), mask)
        ok += 1

    print(f" {dataset}/{split}: {ok}/{len(od_params)} images to zone masks saved")
    print(f"{masks_dir}")


def main():
    for dataset in datasets:
        print(f"\n── {dataset} ──")
        for split in splits:
            process_split(dataset, split)

    print(f"\n✓ All zone masks → {zone_output_root}")
    print("\nZone structure per image:")
    print("  {stem}_zone_i.png     ← Zone I  (0–2r)")
    print("  {stem}_zone_ii.png    ← Zone II (2r–5r)")
    print("  {stem}_ridge.png      ← Ridge   (5r–7r)")
    print("  {stem}_periphery.png  ← Periphery (>7r)")


if __name__ == "__main__":
    main()
