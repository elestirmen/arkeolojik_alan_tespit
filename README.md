Archaeological Feature Detection (DL + Classic)
==============================================

Overview
--------
- This tool detects potential archaeological features from a multi‑band GeoTIFF using two complementary pipelines:
  - Deep learning (DL) segmentation with a U‑Net architecture (via `segmentation_models_pytorch`).
  - Classical, dependency‑light methods that use relief derivatives and image processing (SciPy + RVT).
- Both pipelines run on tiles with optional feather blending for seamless mosaics. Probability rasters and thresholded masks are written as GeoTIFFs; optionally, detections are vectorized to a GeoPackage.

Input Assumptions
-----------------
- Single multi‑band GeoTIFF that aligns RGB, DSM, and DTM onto the same grid/CRS.
- Expected band order: `R,G,B,DSM,DTM`. Configure with `--bands` if different. Use `0` for a missing band (e.g., `DSM=0`).
- A DSM enables tall‑object masking (e.g., trees/buildings) via `--mask-talls`. If DSM is missing (`0`), tall‑masking is disabled.

Channel Stack Used by DL
------------------------
- We derive a 9‑channel stack: `[R, G, B, SVF, PosOpen, NegOpen, LRM, Slope, nDSM]`.
- Relief layers are computed from the DTM using RVT (Relief Visualization Toolbox):
  - SVF (sky‑view factor), positive/negative openness, LRM (local relief model), Slope.
- nDSM is (DSM − DTM). All channels are normalised robustly via 2–98 percentiles (per tile or globally, see `--global-norm`).

Classical Scoring (No New Heavy Deps)
-------------------------------------
Three modes, selectable via `--classic-modes` (comma-separated):
- `rvtlog`: Mixes RVT layers, multi‑scale Laplacian‑of‑Gaussian blobs, gradients, and local variance.
  - Score = 0.30·LoG(σ∈{1,2,4,8}) + 0.20·LRM + 0.15·(1−SVF) + 0.15·∥∇DTM∥ + 0.10·NegOpen + 0.10·LocalVar
- `hessian`: Multi‑scale Hessian (σ∈{1,2,4,8}) using λ₂ response (ridge/valley strength).
- `morph`: Multi‑scale morphological opening/closing (r∈{3,5,9,15}) → white/black top‑hat prominence.

All classic scores are normalized to [0,1] and thresholded. If `--classic-th` is not set, Otsu thresholding is used on the 0..1 scores, respecting nodata.

Fusion (DL + Classic)
---------------------
- If both DL and classic run, you may blend them: `fused = alpha·DL + (1−alpha)·Classic` with `--fuse --alpha 0.5` (default α=0.5).
- Fusion outputs additional `*_fused_prob.tif` and `*_fused_mask.tif`.

Tiling and Feathering
---------------------
- Tiles are extracted as per `--tile` and `--overlap`. Optional cosine feathering blends overlaps (`--feather` by default, disable with `--no-feather`).
- Probability accumulation respects per‑tile valid masks (nodata floats propagate as NaN).

Outputs and Naming
------------------
- DL: `<out>_prob.tif`, `<out>_mask.tif`
- Classic per‑mode (when `--classic-save-intermediate` or only one mode selected): `<out>_classic_{mode}_prob.tif`, `<out>_classic_{mode}_mask.tif`
- Combined classic (mean of selected modes): `<out>_classic_prob.tif`, `<out>_classic_mask.tif`
- Fusion: `<out>_fused_prob.tif`, `<out>_fused_mask.tif`
- Vectorization (if `--vectorize`): `<out>.gpkg` for DL; classic combined and per‑mode (if written) and fused each get their own GPKG.

Installation
------------
1) Create an environment (Python 3.10 recommended for widest wheel availability):
- Conda: `conda create -n archai python=3.10 && conda activate archai`
- venv: `python -m venv .venv && . .venv/Scripts/activate` (Windows PowerShell: `.venv\Scripts\Activate.ps1`)

2) Install dependencies:
- `pip install -r requirements.txt`
- Notes on RVT:
  - For Python < 3.11, `rvt-py` wheels are used.
  - For Python ≥ 3.11, `rvt` is specified. If a wheel is not available for your platform, consider using Python 3.10.

3) GPU (optional):
- Install a CUDA‑enabled PyTorch build if you have an NVIDIA GPU. Check https://pytorch.org/get-started/locally/ for the right `pip` command.

Quick Start
-----------
Minimal run (DL + classic enabled by default):
- `python archaeo_detect.py --input your_multiband.tif --bands 1,2,3,4,5`

Common options:
- `--tile 1024 --overlap 256` control tiling.
- `--th 0.6` DL threshold for binary mask.
- `--mask-talls 2.5` zeroes DL detections where nDSM > 2.5 m.
- `--vectorize` writes polygons (GPKG) per output mask.
- `--out-prefix /path/to/out/siteA` changes output location/prefix.

Zero‑shot DL (no weights; use ImageNet encoder inflated to 9‑ch):
- `python archaeo_detect.py --input site.tif --zero-shot-imagenet`

Classic controls:
- Enable/disable classic: `--classic` (default on), `--no-classic` to turn off.
- Modes: `--classic-modes rvtlog,hessian,morph` or `--classic-modes combo` (shortcut for all three).
- Threshold: `--classic-th 0.5` to override Otsu.
- Save per‑mode rasters too: `--classic-save-intermediate`.

Fusion:
- `--fuse --alpha 0.5` mixes DL and classic probabilities and writes fused rasters (+ optional GPKG).

Examples
--------
- DL + single classic:
  - `python archaeo_detect.py --input site.tif --classic --classic-modes rvtlog`
- DL + multi classic with per‑mode outputs:
  - `python archaeo_detect.py --input site.tif --classic --classic-modes rvtlog,hessian,morph --classic-save-intermediate`
- Zero‑shot + combo + fusion:
  - `python archaeo_detect.py --input site.tif --zero-shot-imagenet --classic --classic-modes combo --fuse --alpha 0.5`

CLI Reference (selected)
------------------------
- `--input`: path to multi‑band GeoTIFF (R,G,B,DSM,DTM).
- `--bands`: band indices CSV, e.g., `1,2,3,4,5`; use `0` for missing DSM.
- `--arch`, `--encoder`: DL model selection (SMP). Defaults: `Unet`, `resnet34`.
- `--weights`: path to `.pth` weights; or use `--zero-shot-imagenet`.
- `--tile`, `--overlap`, `--feather/--no-feather`: tiling and blending.
- `--global-norm`, `--norm-sample-tiles`: use global 2–98 percentiles estimated from sample tiles.
- `--th`: DL threshold.
- `--mask-talls`: disable DL pixels where nDSM > value.
- `--classic/--no-classic`, `--classic-modes`, `--classic-th`, `--classic-save-intermediate`.
- `--fuse`, `--alpha`: DL–classic blending.
- `--vectorize`, `--min-area`, `--simplify`: polygonisation options.

How It Works (DL)
-----------------
1. Read tiles with specified overlap; compute RVT derivatives for each DTM tile.
2. Assemble 9‑channel tensor, normalize (robust percentiles), and run the U‑Net.
3. Feather probabilities into the global map; mask tall objects if DSM present.
4. Threshold to a binary mask and write GeoTIFFs.
5. Optionally vectorize with mean probability per polygon.

How It Works (Classic)
----------------------
1. Read DTM tiles; compute scores per mode (`rvtlog`, `hessian`, `morph`).
2. Normalize scores to [0,1]; accumulate with feathering while respecting nodata.
3. If multiple modes are selected, average their scores to form the combined classic probability.
4. Threshold via Otsu or `--classic-th`, write per‑mode (optional) and combined GeoTIFFs.

Vectorization
-------------
- Connected components label the binary mask; small areas are filtered by `--min-area` (in projected units or source units if unknown).
- Polygons are simplified if `--simplify` is set; mean probability per polygon is stored as an attribute.

Performance Tips
----------------
- GPU DL inference is much faster; install the correct CUDA PyTorch build.
- Increase `--tile` to reduce per‑tile overhead; ensure you have enough RAM.
- Use `--no-feather` to speed up accumulation if seams are acceptable.
- For large rasters, consider `--global-norm` for consistent channel scaling across tiles.

Troubleshooting
---------------
- No classic outputs:
  - Ensure DTM band index is > 0 (`--bands` 5th entry). With `--no-classic` disabled by accident, classic is on by default.
  - Check outputs written next to `--out-prefix` with `_classic_` in names.
- RVT import errors:
  - Prefer Python 3.10; install from wheels with `pip install rvt-py` (or `rvt` on newer Python).
- Raster band order wrong:
  - Use `--bands R,G,B,DSM,DTM`. If DSM is absent, set the 4th to `0` and tall masking is disabled.
- Memory issues:
  - Reduce `--tile` and/or disable `--global-norm`. Use `--no-feather`.

License
-------
- Copyright remains with the repository owner. See project files.

