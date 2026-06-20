"""
preprocessing.py
----------------
Load and preprocess BraTS volumes for inference.

Supports two input modes:
  1. Pre-built .npz files  (same format used during training)
  2. Raw NIfTI files       (one file per MRI modality)

The four expected BraTS modalities and their array order:
  Index 0 → T1
  Index 1 → T1ce
  Index 2 → T2
  Index 3 → FLAIR
"""

import os
import numpy as np
import torch
from typing import Tuple, Optional, List


# ── Normalisation ─────────────────────────────────────────────────────────────
def z_score_normalise(volume: np.ndarray) -> np.ndarray:
    """
    Per-channel Z-score normalisation, computed only over non-zero voxels
    so that the large background region does not skew the statistics.

    Args:
        volume: float32 array of shape (C, D, H, W)

    Returns:
        Normalised array of the same shape.
    """
    out = np.zeros_like(volume, dtype=np.float32)
    for c in range(volume.shape[0]):
        ch   = volume[c]
        mask = ch > 0
        if mask.any():
            mu    = ch[mask].mean()
            sigma = ch[mask].std()
            out[c] = np.where(mask, (ch - mu) / (sigma + 1e-8), 0.0)
        # else: channel stays zero (no signal)
    return out


# ── Padding helpers ───────────────────────────────────────────────────────────
def pad_to_multiple(
    volume: np.ndarray,
    multiple: int = 16,
) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """
    Pad a (C, D, H, W) volume so that every spatial dimension is a multiple
    of `multiple`.  Padding is added at the *end* of each axis.

    Returns:
        padded_volume : padded array
        pad_amounts   : (pd, ph, pw) — amount added to each spatial axis,
                        needed to crop back after inference.
    """
    _, D, H, W = volume.shape
    pad = lambda n: (multiple - n % multiple) % multiple
    pd, ph, pw = pad(D), pad(H), pad(W)
    if pd or ph or pw:
        volume = np.pad(volume, ((0, 0), (0, pd), (0, ph), (0, pw)))
    return volume, (pd, ph, pw)


def unpad(mask: np.ndarray, pad_amounts: Tuple[int, ...]) -> np.ndarray:
    """
    Remove the padding that was added by pad_to_multiple.

    Args:
        mask        : (D', H', W') integer array
        pad_amounts : (pd, ph, pw) as returned by pad_to_multiple

    Returns:
        Cropped mask of shape (D, H, W).
    """
    pd, ph, pw = pad_amounts
    D, H, W = mask.shape
    return mask[
        : D - pd if pd else D,
        : H - ph if ph else H,
        : W - pw if pw else W,
    ]


# ── NPZ loader ────────────────────────────────────────────────────────────────
def load_npz(
    path: str,
    normalise: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load a pre-built .npz file that contains keys ``X`` and optionally ``Y``.

    Args:
        path      : path to the .npz file
        normalise : apply Z-score normalisation to X

    Returns:
        X : float32 (C, D, H, W)
        Y : int64   (D, H, W)  or None if the file has no ground-truth label
    """
    data = np.load(path)
    X    = data["X"].astype(np.float32)
    Y    = data["Y"].astype(np.int64) if "Y" in data else None
    data.close()

    if normalise:
        X = z_score_normalise(X)

    return X, Y


# ── NIfTI loader ──────────────────────────────────────────────────────────────
def load_nifti(
    t1_path:    str,
    t1ce_path:  str,
    t2_path:    str,
    flair_path: str,
    label_path: Optional[str] = None,
    normalise:  bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load four BraTS NIfTI modalities and optionally a segmentation label.

    Requires ``nibabel``.  Install with:  pip install nibabel

    Args:
        t1_path    : path to T1    .nii / .nii.gz
        t1ce_path  : path to T1ce  .nii / .nii.gz
        t2_path    : path to T2    .nii / .nii.gz
        flair_path : path to FLAIR .nii / .nii.gz
        label_path : (optional) path to segmentation .nii / .nii.gz
        normalise  : apply Z-score normalisation to X

    Returns:
        X : float32 (4, D, H, W)
        Y : int64   (D, H, W)  or None
    """
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError(
            "nibabel is required for NIfTI loading.  "
            "Install it with:  pip install nibabel"
        ) from exc

    channels: List[np.ndarray] = []
    for p in [t1_path, t1ce_path, t2_path, flair_path]:
        vol = nib.load(p).get_fdata(dtype=np.float32)
        channels.append(vol)

    # Stack → (C, D, H, W).  NIfTI spatial order is (X, Y, Z); we treat
    # the three spatial axes generically and do not reorder them here.
    X = np.stack(channels, axis=0)

    Y: Optional[np.ndarray] = None
    if label_path is not None:
        Y = nib.load(label_path).get_fdata(dtype=np.float32).astype(np.int64)

    if normalise:
        X = z_score_normalise(X)

    return X, Y


# ── Torch tensor factory ──────────────────────────────────────────────────────
def to_tensor(
    X: np.ndarray,
    device: torch.device,
    pad_multiple: int = 16,
) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """
    Pad and convert a (C, D, H, W) numpy array to a batched float32 tensor.

    Returns:
        tensor      : shape (1, C, D', H', W') on `device`
        pad_amounts : passed to unpad() after inference
    """
    X, pad_amounts = pad_to_multiple(X, multiple=pad_multiple)
    tensor = torch.from_numpy(X).unsqueeze(0).to(device, non_blocking=True)
    return tensor, pad_amounts