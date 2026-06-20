"""
postprocessing.py
-----------------
Post-processing utilities applied to the raw segmentation mask, and
optional evaluation metrics when ground-truth labels are available.

BraTS class conventions
-----------------------
  0 → Background
  1 → Necrotic / Non-enhancing Tumour Core  (NCR/NET)
  2 → Peritumoral Oedema                    (ED)
  3 → Enhancing Tumour                      (ET)

BraTS composite regions
-----------------------
  Whole Tumour   (WT)  = classes {1, 2, 3}
  Tumour Core    (TC)  = classes {1, 3}
  Enhancing Tumour (ET) = class  {3}
"""

import numpy as np
from typing import Dict, List, Optional

try:
    from scipy.ndimage import label as scipy_label
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# ── Label-map clean-up ────────────────────────────────────────────────────────
def remove_small_components(
    mask:          np.ndarray,
    min_voxels:    int = 64,
    background:    int = 0,
) -> np.ndarray:
    """
    Remove isolated connected components smaller than `min_voxels` voxels.

    Requires scipy.  If scipy is not installed the mask is returned unchanged
    with a warning printed to stdout.

    Args:
        mask       : (D, H, W) integer label array
        min_voxels : components with fewer voxels than this are zeroed out
        background : label value treated as background (never removed)

    Returns:
        Cleaned mask of the same shape and dtype.
    """
    if not _SCIPY_AVAILABLE:
        print("[postprocessing] scipy not available — skipping small-component removal")
        return mask

    out = mask.copy()
    for cls in np.unique(mask):
        if cls == background:
            continue
        binary     = (mask == cls).astype(np.uint8)
        labelled, n_comp = scipy_label(binary)
        for comp_id in range(1, n_comp + 1):
            comp_mask = labelled == comp_id
            if comp_mask.sum() < min_voxels:
                out[comp_mask] = background
    return out


# ── Dice coefficient ─────────────────────────────────────────────────────────
def dice_coefficient(
    pred:   np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-5,
) -> float:
    """Binary Dice between two boolean / 0-1 arrays."""
    pred   = pred.astype(bool)
    target = target.astype(bool)
    inter  = (pred & target).sum()
    union  = pred.sum() + target.sum()
    return (2 * inter + smooth) / (union + smooth)


def hausdorff_distance(
    pred:   np.ndarray,
    target: np.ndarray,
    percentile: float = 95.0,
) -> float:
    """
    Percentile Hausdorff distance (HD95) between two binary masks.

    Requires scipy.  Returns NaN if scipy is unavailable or either mask
    is empty.
    """
    if not _SCIPY_AVAILABLE:
        return float("nan")

    from scipy.ndimage import distance_transform_edt

    pred   = pred.astype(bool)
    target = target.astype(bool)

    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return float("nan")

    dist_pred   = distance_transform_edt(~pred)
    dist_target = distance_transform_edt(~target)

    hd_pred_to_target = np.percentile(dist_pred[target],   percentile)
    hd_target_to_pred = np.percentile(dist_target[pred],   percentile)

    return float(max(hd_pred_to_target, hd_target_to_pred))


# ── Per-class and composite metrics ──────────────────────────────────────────
def compute_metrics(
    pred:        np.ndarray,
    target:      np.ndarray,
    num_classes: int = 4,
    compute_hd:  bool = False,
) -> Dict[str, float]:
    """
    Compute a full suite of segmentation metrics.

    Per-class metrics (Dice, HD95 if requested):
        "dice_cls_{c}"   for c in 0 … num_classes-1
        "hd95_cls_{c}"   for c in 0 … num_classes-1  (only when compute_hd)

    BraTS composite region metrics:
        "dice_WT", "dice_TC", "dice_ET"
        "hd95_WT", "hd95_TC", "hd95_ET"   (only when compute_hd)

    Summary:
        "mean_dice"     mean over foreground classes (1, 2, 3)
        "mean_iou"      mean over foreground classes

    Args:
        pred        : (D, H, W) int prediction
        target      : (D, H, W) int ground truth
        num_classes : number of label classes
        compute_hd  : also compute HD95 (slow — requires scipy)

    Returns:
        Dictionary of metric_name → float.
    """
    metrics: Dict[str, float] = {}

    # ── Per-class Dice / IoU ──────────────────────────────────────────────────
    dice_fg: List[float] = []
    iou_fg:  List[float] = []

    for c in range(num_classes):
        p_c = pred   == c
        t_c = target == c
        d   = dice_coefficient(p_c, t_c)
        metrics[f"dice_cls_{c}"] = d

        # IoU
        inter = (p_c & t_c).sum()
        union = (p_c | t_c).sum()
        metrics[f"iou_cls_{c}"] = float((inter + 1e-5) / (union + 1e-5))

        if compute_hd:
            metrics[f"hd95_cls_{c}"] = hausdorff_distance(p_c, t_c)

        if c > 0:                               # exclude background from mean
            dice_fg.append(d)
            iou_fg.append(metrics[f"iou_cls_{c}"])

    metrics["mean_dice"] = float(np.mean(dice_fg)) if dice_fg else 0.0
    metrics["mean_iou"]  = float(np.mean(iou_fg))  if iou_fg  else 0.0

    # ── BraTS composite regions ───────────────────────────────────────────────
    regions = {
        "WT": np.isin(pred, [1, 2, 3]),   # Whole Tumour
        "TC": np.isin(pred, [1, 3]),       # Tumour Core
        "ET": pred == 3,                   # Enhancing Tumour
    }
    gt_regions = {
        "WT": np.isin(target, [1, 2, 3]),
        "TC": np.isin(target, [1, 3]),
        "ET": target == 3,
    }
    for name in ("WT", "TC", "ET"):
        d = dice_coefficient(regions[name], gt_regions[name])
        metrics[f"dice_{name}"] = d
        if compute_hd:
            metrics[f"hd95_{name}"] = hausdorff_distance(
                regions[name], gt_regions[name]
            )

    return metrics


# ── Pretty-print ──────────────────────────────────────────────────────────────
def print_metrics(metrics: Dict[str, float]) -> None:
    """Print a formatted summary of the metrics dictionary."""
    print("\n" + "─" * 50)
    print("  Segmentation Metrics")
    print("─" * 50)

    # Per-class Dice
    print("  Per-class Dice:")
    cls_labels = {0: "Background", 1: "NCR/NET", 2: "Oedema", 3: "Enh. Tumour"}
    for c, label in cls_labels.items():
        key = f"dice_cls_{c}"
        if key in metrics:
            print(f"    Class {c} ({label:<14s}): {metrics[key]:.4f}")

    # BraTS composite
    print("  BraTS Composite Dice:")
    for region in ("WT", "TC", "ET"):
        key = f"dice_{region}"
        if key in metrics:
            print(f"    {region}: {metrics[key]:.4f}")

    # HD95 (if computed)
    hd_keys = [k for k in metrics if k.startswith("hd95")]
    if hd_keys:
        print("  HD95:")
        for k in sorted(hd_keys):
            print(f"    {k}: {metrics[k]:.2f}")

    print(f"\n  Mean Dice (FG): {metrics.get('mean_dice', float('nan')):.4f}")
    print(f"  Mean IoU  (FG): {metrics.get('mean_iou',  float('nan')):.4f}")
    print("─" * 50 + "\n")