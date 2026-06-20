<div align="center">

# 🧠 M1 — 3D Attention Res-U-Net for Glioma Segmentation

### The Entry Point of NeuroSight

![Val Dice](https://img.shields.io/badge/Val%20Dice-0.8851-2ea44f?style=for-the-badge)
![Params](https://img.shields.io/badge/Params-34.17M-blue?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-BraTS%202024-9b59b6?style=for-the-badge)
![Framework](https://img.shields.io/badge/PyTorch-DDP%20%2B%20AMP-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Training%20Complete-success?style=for-the-badge)

*Part of [NeuroSight](#) — end-to-end clinical AI for GBM treatment monitoring and early detection.*

</div>

---

## 🩻 Why MRI Comes First

Every NeuroSight patient enters the system through one of two doors: an MRI scan, or a blood draw for cfDNA. Almost everyone walks through the MRI door first.

| | MRI Path (M1 → M2 → M3) | cfDNA Path (M5) |
|---|---|---|
| **Already standard of care** | Yes — ordered for nearly every suspected brain tumor | No — requires a dedicated liquid biopsy workflow |
| **Invasiveness** | Non-invasive | Blood draw only, but lab-dependent |
| **Infrastructure required** | Any hospital with an MRI scanner | Specialized sequencing pipeline |
| **Turnaround** | Same day | Days to weeks |
| **What it answers** | "Where is the tumor, and how is it changing?" | "Is there tumor-derived signal in plasma at all?" |

Because of this, **M1 is where NeuroSight starts for almost every real patient.** It is the first model in the pipeline, the first decision node in the LangGraph agent, and the model every other component depends on, directly or indirectly. M2's physics-informed growth simulation needs M1's segmentation as its initial boundary condition. M3's progression classifier needs two M1 segmentations (baseline and follow-up) to compute its delta features. M4's structured clinical report pulls six summary fields computed directly from M1's output mask. Get M1 wrong, and everything downstream inherits the error.

This repo is that first model.

---

## 📊 At a Glance

| | |
|---|---|
| **Task** | Multi-class 3D tumor sub-region segmentation |
| **Input** | 4-channel MRI volume (T1, T1ce, T2, FLAIR), 96×96×96 patches |
| **Output** | 4-class voxel-wise mask (background, necrotic core, edema, enhancing tumor) |
| **Architecture** | 3D Res-U-Net + attention-gated skip connections + deep supervision |
| **Parameters** | 34.17M |
| **Dataset** | BraTS 2024, ~200 patient volumes |
| **Best Validation Dice** | **0.8851** (epoch 84 / 100) |
| **Hardware** | 2× Kaggle T4, DistributedDataParallel |
| **Training time** | ~46s/epoch → ~76 minutes total |
| **Framework** | PyTorch, AMP mixed precision, NCCL backend |

---

## 🏗️ Architecture

```
Input (4, 96, 96, 96)
  │
  ├─ Encoder 1 (24ch)  ──skip──────────────────┐
  │     ↓ pool                                  │
  ├─ Encoder 2 (48ch)  ──skip──────────────┐    │
  │     ↓ pool                              │    │
  ├─ Encoder 3 (96ch)  ──skip──────────┐    │    │
  │     ↓ pool                          │    │    │
  ├─ Encoder 4 (192ch) ──skip──────┐    │    │    │
  │     ↓ pool                      │    │    │    │
  ├─ Bridge (384ch, 3× ResBlock)    │    │    │    │
  │     ↓ upsample                  │    │    │    │
  ├─ Decoder 4 (192ch) ←─attn gate ─┘    │    │    │ ──ds4 head (aux loss × 0.4)
  │     ↓ upsample                       │    │    │
  ├─ Decoder 3 (96ch)  ←─attn gate ──────┘    │    │ ──ds3 head (aux loss × 0.2)
  │     ↓ upsample                            │    │
  ├─ Decoder 2 (48ch)  ←─attn gate ───────────┘    │ ──ds2 head (aux loss × 0.1)
  │     ↓ upsample                                 │
  ├─ Decoder 1 (24ch)  ←─attn gate ────────────────┘
  │
  └─ Output head (4 classes) ── main loss × 1.0
```

| Design choice | Why |
|---|---|
| **Residual blocks** everywhere (He et al., 2016) | Clean gradient flow through 5 levels of downsampling |
| **Attention gates** at every decoder stage (Oktay et al., 2018) | Suppress irrelevant skip-connection signal before concatenation — biggest win at tumor boundaries, where naive skips leak healthy-tissue noise |
| **Deep supervision**, 3 auxiliary heads | Forces a sane coarse segmentation early in the decoder; visibly stabilizes the first ~20 epochs |
| **GroupNorm**, not BatchNorm | Per-GPU batch size is 1 (3D volumetric memory cost) — BatchNorm statistics would be meaningless at that batch size |

### BraTS label convention

| Label | Class |
|---|---|
| 0 | Background |
| 1 | Necrotic core (NCR) |
| 2 | Edema (ED) |
| 3 | Enhancing tumor (ET) |

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|---|---|
| Crop size | 96 × 96 × 96 |
| Base filter width | 24 |
| Batch size (per GPU) | 1 |
| Gradient accumulation | 4 steps |
| Effective batch size | 8 (1 × 4 × 2 GPUs) |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| LR schedule | Cosine annealing (T_max=100, eta_min=1e-6) |
| Gradient clipping | max norm 1.0 |
| Mixed precision | AMP enabled |
| Epochs | 100 |
| Seed | 42 |
| Foreground crop probability | 0.8 |

**Loss:**

```
CombinedLoss   = 0.5 · CrossEntropy + 0.5 · DiceLoss
Total (train)  = 1.0·loss(main) + 0.4·loss(ds4) + 0.2·loss(ds3) + 0.1·loss(ds2)
```

### Data pipeline

| Component | What it does |
|---|---|
| `StreamingBraTSDataset` | Streams one `.npz` patient volume at a time (image-load is the bottleneck at this scale, not GPU compute) |
| `ForegroundCrop3D` | 80% of crops are centered on a random tumor voxel; 20% are uniform random — tumor tissue is a small fraction of total brain volume, so naive cropping wastes most gradient steps on empty patches |
| `RandomFlip3D` | Independent random flip per spatial axis, training split only |
| DDP data split | Each GPU rank gets a disjoint ~100-patient slice, then an 85/15 train/val split within that slice |

---

## 📈 Final Results

| Epoch | Train Dice | Val Dice | Train IoU | Val IoU | Note |
|---|---|---|---|---|---|
| 1 | 0.4282 | 0.5794 | 0.2857 | 0.3877 | First checkpoint |
| 10 | 0.7666 | 0.7632 | 0.5577 | 0.5664 | |
| 25 | 0.7878 | 0.7846 | 0.5838 | 0.5961 | |
| 50 | 0.8454 | 0.8368 | 0.6441 | 0.6595 | |
| 71 | 0.8568 | 0.8605 | 0.6581 | 0.6688 | |
| **84** | **0.8759** | **0.8851** | **0.6839** | **0.6873** | **Best checkpoint** |
| 100 | 0.8788 | 0.8649 | 0.6903 | 0.6655 | Final epoch (not best) |

IoU (Intersection over Union, the Jaccard index) is the harsher companion metric to Dice — it's included alongside Dice rather than reported alone, since the two together are the standard pairing in segmentation literature and showing both is more honest than leading with whichever number looks better. Val IoU at the best checkpoint (0.6873) sits meaningfully below Val Dice (0.8851) at every epoch, which is expected — Dice and IoU are related by `Dice = 2·IoU / (1+IoU)`, so Dice is mathematically always the larger number for the same underlying overlap. They won't satisfy that exact formula here since this implementation aggregates them slightly differently (per-class Dice averaged vs. sklearn's macro-average `jaccard_score` for IoU), but they move together and tell the same story.

> **Voxel accuracy (not reported in the table above) stayed at 99%+ from epoch 1 onward**, which is exactly why it isn't tracked as a metric here. Background occupies the overwhelming majority of any brain MRI volume, so a model predicting "background everywhere" would already score 95%+ accuracy on day one without having learned anything about the tumor — epoch 1's train accuracy of 0.9357, recorded before the model could segment anything meaningfully, makes that plain. Accuracy is the wrong lens for this task; Dice and IoU are what actually measure whether the small, clinically relevant tumor region was found.

> **A caveat worth stating plainly:** the Dice and IoU above are a 4-class macro average **including background**, not the standard BraTS WT/TC/ET (Whole Tumor / Tumor Core / Enhancing Tumor) binary-region convention that published leaderboards report. Background scores on mostly-empty volumes sit near 1.0 and pull both macro averages up. The 0.8851 Dice is real, but a WT/TC/ET re-evaluation is the right next step before quoting this number against other BraTS results externally.

---

## 🔗 Role in NeuroSight

| Stage | Model | Role | Consumes M1's output? |
|---|---|---|---|
| **M1** | **Res-U-Net (this repo)** | **Segments tumor sub-regions from a single MRI** | — |
| M2 | Fisher-KPP PINN | Simulates patient-specific tumor growth forward in time | ✅ Uses `pred_mask.npy` as the initial boundary condition |
| M3 | XGBoost + MLP | Classifies true progression vs. pseudoprogression from two scans | ✅ Needs M1 run on both baseline and follow-up scans |
| M4 | RAG report generator | Produces the structured clinical research brief | ✅ Pulls 6 summary fields (volumes, centroids) computed directly from the mask |
| NeuroBio Agent | LangGraph reasoning agent | Forms biological hypotheses from upstream model outputs | ✅ Indirectly, via M2/M3/M4 outputs |

---

## 🔧 Engineering Notes

| Issue / Decision | Detail |
|---|---|
| GroupNorm channel safety | Helper picks the largest valid group `[32,16,8,4,2,1]` that evenly divides channel count, so it never errors on deep-supervision heads or the 1-channel attention output |
| `\b` backspace bug | A literal backspace character had been hardcoded into the M1 checkpoint path string during an earlier integration pass — silently broke path resolution, caught and fixed before this training run |
| Deep supervision weights (1.0/0.4/0.2/0.1) | Decaying weights, not equal — keeps the final full-resolution output dominant while still giving real gradient signal to coarser heads (standard nnU-Net-style pattern) |
| Foreground-biased cropping | Empirically necessary, not optional — uniform cropping wasted most steps on tumor-free patches given how little of a brain volume tumor occupies |

---

## ⚠️ Limitations

| # | Limitation | Why it matters | Next step |
|---|---|---|---|
| 1 | Reported Dice (0.8851) is 4-class macro incl. background | Not directly comparable to published BraTS WT/TC/ET numbers | Re-evaluate with standard binary region groupings |
| 2 | No held-out test set | Validation set was also used for checkpoint selection — some optimistic bias | Carve out a third, fully untouched split |
| 3 | No sliding-window / TTA inference | Current inference is patch-based at training crop size | Implement full-volume sliding-window inference |
| 4 | Raw voxel accuracy is logged but not reported | Sits at 99%+ from epoch 1, before the model has learned anything — background dominates total volume, making accuracy uninformative for this task | Not a model flaw — a metric-choice note. Dice/IoU are reported instead for exactly this reason |

---

## 📁 Repository Structure

```
.
├── train_worker.py          # Model, loss, dataset, augmentation, DDP training loop
├── train_launch.py          # DDP entry point (mp.spawn across available GPUs)
├── checkpoints/
│   └── best_model.pth       # Best validation Dice checkpoint
└── README.md
```

---

## 📚 References

- Çiçek, Ö., et al. *3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation.* MICCAI 2016.
- He, K., et al. *Deep Residual Learning for Image Recognition.* CVPR 2016.
- Oktay, O., et al. *Attention U-Net: Learning Where to Look for the Pancreas.* MIDL 2018.
- Wu, Y., & He, K. *Group Normalization.* ECCV 2018.
- BraTS 2024 Challenge — Brain Tumor Segmentation, RSNA-ASNR-MICCAI.

---

<div align="center">

**M1** (this repo) → [M2 — Fisher-KPP Growth PINN](#) → [M3 — Progression Classifier](#) → [M4 — RAG Clinical Reporting](#)

*reasoned over by the [NeuroBio Agent](#)*

</div>
