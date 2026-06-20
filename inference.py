"""
inferencer.py
-------------
Model loading and prediction logic.

Two inference strategies are provided:

  full_volume_inference
      Runs the full padded volume through the model in a single forward pass.
      Fast, but may OOM on large volumes or small GPUs.

  sliding_window_inference
      Tiles the volume into overlapping 3-D patches, runs each patch through
      the model, and Gaussian-blends the overlapping predictions back into a
      single probability map.  More memory-efficient and usually gives
      slightly better boundary accuracy due to the importance weighting.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple

from model import ResUNet3D, IN_CHANNELS, NUM_CLASSES, BASE_FILTERS


# ── Checkpoint loading ────────────────────────────────────────────────────────
def load_model(
    checkpoint_path: str,
    device: torch.device,
    in_channels:  int = IN_CHANNELS,
    num_classes:  int = NUM_CLASSES,
    base_filters: int = BASE_FILTERS,
) -> ResUNet3D:
    """
    Instantiate ResUNet3D and load weights from a training checkpoint.

    The checkpoint is expected to be the dict saved by the trainer, i.e.:
        {
            "epoch":      int,
            "state_dict": OrderedDict,   # model.module.state_dict()
            "optimizer":  ...,
            "scheduler":  ...,
            "val_dice":   float,
            "history":    dict,
        }

    Args:
        checkpoint_path : path to the .pth file
        device          : torch device to load weights onto
        in_channels     : must match training value
        num_classes     : must match training value
        base_filters    : must match training value

    Returns:
        model in eval() mode, on `device`
    """
    model = ResUNet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        base_filters=base_filters,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Accept bare state-dicts as well as full checkpoint dicts
    state_dict = ckpt.get("state_dict", ckpt)

    # Strip "module." prefix produced by DDP wrapping
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(cleaned, strict=True)
    if missing:
        raise RuntimeError(f"Missing keys in checkpoint: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected}")

    model.to(device).eval()

    epoch    = ckpt.get("epoch",    "?")
    val_dice = ckpt.get("val_dice", float("nan"))
    print(f"[inferencer] Loaded checkpoint — epoch {epoch}, "
          f"best val Dice {val_dice:.4f}")

    return model


# ── Gaussian importance kernel ────────────────────────────────────────────────
def _gaussian_kernel(size: Tuple[int, int, int], sigma_scale: float = 0.125) -> torch.Tensor:
    """
    Build a 3-D Gaussian importance map for a patch of the given `size`.

    Voxels near the centre of the patch receive higher weight, smoothly
    down-weighting predictions near patch edges where context is limited.

    Args:
        size        : (D, H, W) patch dimensions
        sigma_scale : sigma = sigma_scale * dimension for each axis

    Returns:
        Float32 tensor of shape (1, 1, D, H, W), values in (0, 1].
    """
    def gauss1d(n: int) -> np.ndarray:
        sigma = n * sigma_scale
        x     = np.arange(n) - (n - 1) / 2.0
        g     = np.exp(-0.5 * (x / sigma) ** 2)
        return g / g.max()

    d_kern = gauss1d(size[0])
    h_kern = gauss1d(size[1])
    w_kern = gauss1d(size[2])

    kernel = (
        d_kern[:, None, None]
        * h_kern[None, :, None]
        * w_kern[None, None, :]
    ).astype(np.float32)

    return torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)


# ── Full-volume inference ─────────────────────────────────────────────────────
@torch.no_grad()
def full_volume_inference(
    model:  ResUNet3D,
    tensor: torch.Tensor,
    amp:    bool = True,
) -> np.ndarray:
    """
    Single forward pass over the entire padded volume.

    Args:
        model  : ResUNet3D in eval() mode
        tensor : (1, C, D, H, W) float32 tensor on the model's device
        amp    : use bfloat16/float16 autocast for faster inference

    Returns:
        segmentation mask (D, H, W) int64 numpy array
    """
    device = next(model.parameters()).device
    with torch.autocast(device.type, enabled=amp):
        logits = model(tensor)                          # (1, C, D, H, W)
    mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)
    return mask


# ── Sliding-window inference ──────────────────────────────────────────────────
@torch.no_grad()
def sliding_window_inference(
    model:      ResUNet3D,
    tensor:     torch.Tensor,
    patch_size: Tuple[int, int, int] = (96, 96, 96),
    overlap:    float = 0.5,
    amp:        bool  = True,
) -> np.ndarray:
    """
    Gaussian-blended sliding-window inference.

    Tiles the volume with `overlap` fraction of overlap along every axis,
    runs each tile through the model, and accumulates class probabilities
    weighted by the Gaussian importance kernel.

    Args:
        model      : ResUNet3D in eval() mode
        tensor     : (1, C, D, H, W) float32 on the model's device
        patch_size : (pd, ph, pw) patch dimensions — should match training crop
        overlap    : fraction of each patch dimension that overlaps neighbours
        amp        : autocast

    Returns:
        segmentation mask (D, H, W) int64 numpy array
    """
    device    = next(model.parameters()).device
    _, C, D, H, W = tensor.shape
    pd, ph, pw    = patch_size

    stride_d = max(1, int(pd * (1 - overlap)))
    stride_h = max(1, int(ph * (1 - overlap)))
    stride_w = max(1, int(pw * (1 - overlap)))

    num_classes = model.head.out_channels  # type: ignore[attr-defined]
    accum  = torch.zeros((num_classes, D, H, W), dtype=torch.float32, device=device)
    weight = torch.zeros((1,           D, H, W), dtype=torch.float32, device=device)

    kernel = _gaussian_kernel(patch_size).to(device)  # (1,1,pd,ph,pw)

    # Build start indices — always include one patch that reaches the far edge
    def starts(total: int, patch: int, stride: int):
        idx = list(range(0, total - patch + 1, stride))
        if not idx or idx[-1] + patch < total:
            idx.append(max(total - patch, 0))
        return idx

    d_starts = starts(D, pd, stride_d)
    h_starts = starts(H, ph, stride_h)
    w_starts = starts(W, pw, stride_w)

    total_patches = len(d_starts) * len(h_starts) * len(w_starts)
    processed     = 0

    for ds in d_starts:
        for hs in h_starts:
            for ws in w_starts:
                patch = tensor[
                    :, :,
                    ds : ds + pd,
                    hs : hs + ph,
                    ws : ws + pw,
                ]

                # Pad patch to exact size if near the volume boundary
                actual = patch.shape[2:]
                if actual != (pd, ph, pw):
                    ep = [0, pw - actual[2], 0, ph - actual[1], 0, pd - actual[0]]
                    patch = F.pad(patch, ep)

                with torch.autocast(device.type, enabled=amp):
                    logits = model(patch)               # (1, num_classes, pd, ph, pw)

                # Trim back to actual patch size if padding was applied
                if actual != (pd, ph, pw):
                    logits = logits[:, :, :actual[0], :actual[1], :actual[2]]
                    k      = kernel[:, :, :actual[0], :actual[1], :actual[2]]
                else:
                    k = kernel

                probs = torch.softmax(logits, dim=1).squeeze(0)   # (C, pd, ph, pw)

                accum[
                    :,
                    ds : ds + actual[0],
                    hs : hs + actual[1],
                    ws : ws + actual[2],
                ] += probs * k.squeeze(0)

                weight[
                    :,
                    ds : ds + actual[0],
                    hs : hs + actual[1],
                    ws : ws + actual[2],
                ] += k.squeeze(0)

                processed += 1
                if processed % 10 == 0 or processed == total_patches:
                    print(f"  [sliding window] {processed}/{total_patches} patches", end="\r")

    print()  # newline after the progress counter
    accum /= weight.clamp(min=1e-8)
    mask   = accum.argmax(dim=0).cpu().numpy().astype(np.int64)
    return mask