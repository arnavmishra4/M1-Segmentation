"""
main.py
-------
BraTS 3D Res-UNet — Inference Entry Point

Usage:
    from main import main
    main(
        t1_path    = "BraTS_001_t1.nii.gz",
        t1ce_path  = "BraTS_001_t1ce.nii.gz",
        t2_path    = "BraTS_001_t2.nii.gz",
        flair_path = "BraTS_001_flair.nii.gz",
    )
"""

import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from preprocessing  import load_nifti, to_tensor, unpad
from inference    import load_model, full_volume_inference, sliding_window_inference
from postprocessing import remove_small_components, compute_metrics, print_metrics
from utils          import save_mask_npy, save_mask_nifti, visualise_slice, print_volume_stats


def main(
    t1_path:    str,
    t1ce_path:  str,
    t2_path:    str,
    flair_path: str,
    # ── optional ──────────────────────────────────────────────────────────────
    label_path: str   = None,
    checkpoint: str   = r"NeuroSight\Model 1 (3dUnet)\best_model.pth",
    out_dir:    str   = "results",
    mode:       str   = "sliding_window",   # "sliding_window" | "full_volume"
    save_nifti: bool  = True,
    visualise:  bool  = True,
) -> np.ndarray:
    """
    Run inference on one BraTS patient.

    Args:
        t1_path    : path to T1    .nii / .nii.gz
        t1ce_path  : path to T1ce  .nii / .nii.gz
        t2_path    : path to T2    .nii / .nii.gz
        flair_path : path to FLAIR .nii / .nii.gz
        label_path : (optional) ground-truth segmentation for evaluation
        checkpoint : path to best_model.pth
        out_dir    : folder where outputs are saved
        mode       : "sliding_window" (memory-efficient) or "full_volume" (fast)
        save_nifti : also save mask as .nii.gz
        visualise  : save axial-slice PNG overlay

    Returns:
        pred_mask : (D, H, W) int64 numpy array
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp    = device.type == "cuda"

    model = load_model(checkpoint, device)

    print("\nLoading NIfTI files...")
    X, Y = load_nifti(
        t1_path    = t1_path,
        t1ce_path  = t1ce_path,
        t2_path    = t2_path,
        flair_path = flair_path,
        label_path = label_path,
        normalise  = True,
    )

    name    = os.path.basename(t1_path).replace(".nii.gz", "").replace(".nii", "").replace("_t1", "")
    out_dir = os.path.join(out_dir, name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'═'*55}")
    print(f"  Patient : {name}")
    print(f"  Volume  : {X.shape}  dtype={X.dtype}")
    print(f"  Device  : {device}  |  AMP: {amp}  |  Mode: {mode}")
    print(f"{'═'*55}")

    tensor, pad_amounts = to_tensor(X, device, pad_multiple=16)

    t0 = time.time()
    if mode == "sliding_window":
        raw_mask = sliding_window_inference(model=model, tensor=tensor)
    else:
        raw_mask = full_volume_inference(model, tensor, amp=amp)
    print(f"  Inference time : {time.time() - t0:.1f}s")

    mask = unpad(raw_mask, pad_amounts)
    mask = remove_small_components(mask)

    print_volume_stats(mask)

    if Y is not None:
        Y_eval  = Y[: mask.shape[0], : mask.shape[1], : mask.shape[2]]
        metrics = compute_metrics(mask, Y_eval)
        print_metrics(metrics)
        np.save(os.path.join(out_dir, "metrics.npy"), metrics)

    save_mask_npy(mask, os.path.join(out_dir, "pred_mask.npy"))

    if save_nifti:
        save_mask_nifti(mask, os.path.join(out_dir, "pred_mask.nii.gz"), reference=t1_path)

    if visualise:
        visualise_slice(
            volume   = X,
            mask     = mask,
            out_path = os.path.join(out_dir, "overlay.png"),
            gt_mask  = Y,
        )

    print("\nDone.\n")
    return mask