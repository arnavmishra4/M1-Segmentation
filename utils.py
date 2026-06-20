"""
utils.py
--------
Output-saving and basic visualisation helpers.

Saving formats
--------------
  .npy          raw numpy int64 mask  (always available)
  .nii.gz       NIfTI mask            (requires nibabel)
  .png          axial slice overlay   (requires matplotlib)
"""

import os
import numpy as np
from typing import Optional, Dict


# ── Mask saving ───────────────────────────────────────────────────────────────
def save_mask_npy(mask: np.ndarray, out_path: str) -> None:
    """Save segmentation mask as a .npy file."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.save(out_path, mask.astype(np.int64))
    print(f"[save] mask → {out_path}")


def save_mask_nifti(
    mask:        np.ndarray,
    out_path:    str,
    reference:   Optional[str] = None,
) -> None:
    """
    Save segmentation mask as a NIfTI file, optionally inheriting the
    affine/header from a reference NIfTI (e.g. the input T1 volume).

    Args:
        mask      : (D, H, W) int array
        out_path  : destination path (should end in .nii or .nii.gz)
        reference : path to an existing NIfTI whose affine/header to copy
    """
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError(
            "nibabel is required for NIfTI saving.  "
            "Install it with:  pip install nibabel"
        ) from exc

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    if reference is not None:
        ref  = nib.load(reference)
        img  = nib.Nifti1Image(mask.astype(np.int16), ref.affine, ref.header)
    else:
        img  = nib.Nifti1Image(mask.astype(np.int16), np.eye(4))

    nib.save(img, out_path)
    print(f"[save] mask → {out_path}")


# ── Slice visualisation ───────────────────────────────────────────────────────
# Colour map: index → RGBA  (background is transparent)
_CLASS_COLORS = {
    0: (0.00, 0.00, 0.00, 0.00),   # Background  — transparent
    1: (0.80, 0.10, 0.10, 0.65),   # NCR/NET     — red
    2: (0.10, 0.65, 0.10, 0.65),   # Oedema      — green
    3: (0.10, 0.10, 0.90, 0.65),   # Enh. Tumour — blue
}


def visualise_slice(
    volume:     np.ndarray,
    mask:       np.ndarray,
    slice_idx:  Optional[int]  = None,
    modality:   int            = 1,          # 0=T1, 1=T1ce, 2=T2, 3=FLAIR
    out_path:   Optional[str]  = None,
    gt_mask:    Optional[np.ndarray] = None,
) -> None:
    """
    Display (and optionally save) an axial slice of the MRI with the
    predicted segmentation overlaid.  If `gt_mask` is provided, a second
    column shows the ground-truth overlay for comparison.

    Args:
        volume    : (C, D, H, W) float32 input volume (pre-normalised)
        mask      : (D, H, W) int predicted segmentation
        slice_idx : axial slice index; defaults to the middle slice
        modality  : channel index to display as the greyscale background
        out_path  : if set, save figure to this path instead of showing it
        gt_mask   : (D, H, W) optional ground-truth label for side-by-side
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualisation.  "
            "Install it with:  pip install matplotlib"
        ) from exc

    D = volume.shape[1]
    if slice_idx is None:
        slice_idx = D // 2
    slice_idx = int(np.clip(slice_idx, 0, D - 1))

    mri_slice  = volume[modality, slice_idx]          # (H, W)
    pred_slice = mask[slice_idx]                      # (H, W)

    def make_overlay(seg_slice: np.ndarray) -> np.ndarray:
        """Convert a 2-D label map to an RGBA overlay image."""
        H, W  = seg_slice.shape
        rgba  = np.zeros((H, W, 4), dtype=np.float32)
        for cls, color in _CLASS_COLORS.items():
            m = seg_slice == cls
            rgba[m] = color
        return rgba

    n_cols  = 2 if gt_mask is not None else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
    if n_cols == 1:
        axes = [axes]

    vmin, vmax = mri_slice.min(), mri_slice.max()

    # ── Prediction ────────────────────────────────────────────────────────────
    axes[0].imshow(mri_slice, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].imshow(make_overlay(pred_slice), interpolation="nearest")
    axes[0].set_title(f"Prediction  (slice {slice_idx})", fontsize=11)
    axes[0].axis("off")

    # ── Ground truth (optional) ───────────────────────────────────────────────
    if gt_mask is not None:
        gt_slice = gt_mask[slice_idx]
        axes[1].imshow(mri_slice, cmap="gray", vmin=vmin, vmax=vmax)
        axes[1].imshow(make_overlay(gt_slice), interpolation="nearest")
        axes[1].set_title(f"Ground Truth (slice {slice_idx})", fontsize=11)
        axes[1].axis("off")

    # ── Legend ────────────────────────────────────────────────────────────────
    labels = {1: "NCR/NET", 2: "Oedema", 3: "Enh. Tumour"}
    patches = [
        mpatches.Patch(facecolor=_CLASS_COLORS[c][:3], label=lbl, alpha=0.7)
        for c, lbl in labels.items()
    ]
    fig.legend(
        handles=patches, loc="lower center",
        ncol=len(patches), fontsize=9, frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )

    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[visualise] figure → {out_path}")
        plt.close(fig)
    else:
        plt.show()


# ── Summary printout ──────────────────────────────────────────────────────────
def print_volume_stats(
    mask:       np.ndarray,
    voxel_mm3:  float = 1.0,
) -> None:
    """
    Print per-class voxel counts and (optionally) volumes in mm³.

    Args:
        mask      : (D, H, W) int segmentation
        voxel_mm3 : voxel volume in cubic millimetres (product of voxel spacings)
    """
    cls_names = {0: "Background", 1: "NCR/NET", 2: "Oedema", 3: "Enh. Tumour"}
    print("\n" + "─" * 45)
    print("  Predicted Volume Statistics")
    print("─" * 45)
    total = mask.size
    for c, name in cls_names.items():
        n      = (mask == c).sum()
        frac   = 100.0 * n / total
        vol    = n * voxel_mm3
        print(f"  {name:<18s}: {n:>8d} vox  ({frac:5.1f}%)  {vol:>10.1f} mm³")
    print("─" * 45 + "\n")