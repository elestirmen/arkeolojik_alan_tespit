Archaeological Feature Detection (DL + Classic)
==============================================

Purpose
-------
- Extract likely archaeological features (e.g., mounds, ring ditches, banks, ditches) from high‑resolution multi‑band DEM products.
- Combine complementary approaches:
  - Deep Learning (DL) U‑Net segmentation over a 9‑channel tensor assembled from imagery and relief derivatives.
  - “Classic” methods relying on terrain analysis and image processing only (SciPy + RVT), with no additional heavy ML deps.

Key Features
------------
- Tiled inference with overlap and cosine feathering for seamless mosaics.
- Robust normalization (2–98 percentile) per tile or globally across sampled tiles.
- DL zero‑shot option inflating a 3‑ch ImageNet encoder to 9‑ch (exploratory mode).
- Three classical scorers (rvtlog, hessian, morph) and a combo averaging them.
- Optional DL–Classic fusion with α‑blend.
- GeoTIFF outputs (probability + mask) and optional vectorization to GeoPackage with attributes.
 - Default behavior: both DL and all classic modes (combo) run out‑of‑the‑box; override via CLI or edit defaults in `archaeo_detect.py`.

Data Requirements
-----------------
- Input: single multi‑band GeoTIFF that co‑aligns RGB, DSM, and DTM in the same CRS/extent/pixel grid.
- Band order: `R,G,B,DSM,DTM` (1‑based indices). Configure with `--bands`. Use `0` for missing DSM; classic requires DTM.
- DSM enables tall‑object masking (set `--mask-talls` height in meters). If DSM index is `0`, tall masking is automatically disabled.

Nine‑Channel Tensor (for DL)
----------------------------
- Channels: `[R, G, B, SVF, PosOpen, NegOpen, LRM, Slope, nDSM]`.
- From DTM with RVT we compute: sky‑view factor (SVF), positive/negative openness, local relief model (LRM), and slope.
- nDSM = DSM − DTM. All channels are normalized to [0,1] with percentile clipping (2–98).

Classic Scoring Modes
---------------------
- rvtlog: Multiscale LoG + RVT + gradient + local variance
  - LoG magnitude: max over σ∈{1,2,4,8}.
  - Gradient magnitude: Gaussian σ=1.5.
  - SVF complement: 1 − SVF.
  - LRM and negative openness (normalized).
  - Local variance over 7×7.
  - Score = 0.30·LoG + 0.20·LRM + 0.15·(1−SVF) + 0.15·∥∇DTM∥ + 0.10·NegOpen + 0.10·Var.
- hessian: Multiscale Hessian λ₂ response (σ∈{1,2,4,8}); ridge/valley strength = |λ₂| normalized.
- morph: Morphological white/black top‑hat
  - Opening/closing with sizes r∈{3,5,9,15}.
  - White top‑hat = DTM − opening; black top‑hat = closing − DTM.
  - Take max of normalized WTH/BTH across radii.

Thresholding and Otsu
---------------------
- Classical scores are normalized into [0,1]. If `--classic-th` is given, it is used directly.
- Otherwise, a 256‑bin Otsu threshold is computed over valid pixels only.
- DL masks use `--th` (default from `USER_DEFAULTS`).
- Fusion masks use `--th` if set, otherwise 0.5.

Fusion (DL + Classic)
---------------------
- `p_fused = alpha·p_dl + (1−alpha)·p_classic` with `--fuse --alpha 0.5` (0≤α≤1).
- Produces `*_fused_prob.tif` and `*_fused_mask.tif`.

Tiling and Feathering
---------------------
- Tiles of size `--tile` with `--overlap` are processed sequentially; probabilities are blended into a global mosaic.
- Cosine half‑ramps over the overlap ensure smooth seams; accumulation weights are masked by valid nodata‑aware masks.

Normalization Choices
---------------------
- Per‑tile normalization (default): robust 2–98 percentiles on each tile independently.
- Global normalization (`--global-norm`): compute medians of per‑channel p2/p98 on a sample of tiles (`--norm-sample-tiles`), and apply to all tiles.

Outputs and Naming
------------------
- DL: `<out>_prob.tif`, `<out>_mask.tif`.
- Classic combined: `<out>_classic_prob.tif`, `<out>_classic_mask.tif`.
- Classic per‑mode (if single mode or `--classic-save-intermediate`): `<out>_classic_{mode}_prob.tif`, `<out>_classic_{mode}_mask.tif`.
- Fusion: `<out>_fused_prob.tif`, `<out>_fused_mask.tif`.
- GeoTIFF metadata: prob float32 (nodata=NaN, deflate), mask uint8 (nodata=0).

Vectorization
-------------
- Optional with `--vectorize`. Components are extracted from the mask; area threshold via `--min-area` (in meters if projected CRS, else source units).
- Attributes include `id`, `area_m2`, and `score_mean` (mean probability within polygon).
- If CRS is geographic, areas are computed in EPSG:6933 internally for accuracy and mapped back.

Installation
------------
- Python 3.10 is recommended for best wheel availability (RVT, GDAL/GEOS).
- Create environment:
  - Conda: `conda create -n archai python=3.10 && conda activate archai`
  - venv: `python -m venv .venv && . .venv/Scripts/activate` (Windows) / `source .venv/bin/activate` (Linux/macOS)
- Install deps: `pip install -r requirements.txt`
- PyTorch GPU (optional): follow https://pytorch.org/get-started/locally/ for the correct CUDA wheel.
- RVT notes:
  - On Python < 3.11, `rvt-py` wheels are used.
  - On Python ≥ 3.11, `rvt` is used if available (some platforms may still lack wheels; downgrade to 3.10 if needed).
- Geo stack on Windows:
  - If `geopandas`/`fiona` fail to install (GDAL/PROJ issues), you can skip vectorization or install via conda (`conda install geopandas fiona`).

Quick Start
-----------
- Minimal (DL + Classic enabled):
  - `python archaeo_detect.py --input your_multiband.tif --bands 1,2,3,4,5`
  - By default, classic modes use `combo` (rvtlog+hessian+morph).
- Specify output prefix:
  - `python archaeo_detect.py --input site.tif --out-prefix outputs/siteA`
- Zero‑shot DL (no weights):
  - `python archaeo_detect.py --input site.tif --zero-shot-imagenet`
- Classic modes:
  - Single: `--classic-modes rvtlog`
  - Multiple: `--classic-modes rvtlog,hessian,morph --classic-save-intermediate`
  - Combo shortcut: `--classic-modes combo` (default; equivalent to rvtlog+hessian+morph)
- Fusion:
  - `--fuse --alpha 0.5`
- Turn classic off:
  - `--no-classic`

Multi-Encoder Runs
------------------
- Default (no extra flags):  
  ```bash
  python archaeo_detect.py --input ... --bands 1,2,3,4,5 --zero-shot-imagenet -v
  ```
  Creates `_resnet34`, `_resnet50`, and `_efficientnet-b3` outputs under the chosen prefix.
- Provide per-encoder checkpoints:  
  ```bash
  python archaeo_detect.py --input ... --bands 1,2,3,4,5 \
    --weights-template "C:/.../models/unet_{encoder}_9ch_best.pth" --vectorize -v
  ```
  Each encoder loads its matching weights when available; others fall back to zero-shot.
- Single encoder only:  
  ```bash
  python archaeo_detect.py --input ... --bands 1,2,3,4,5 \
    --encoders none --encoder resnet50 --zero-shot-imagenet -v
  ```
  Disables the multi-run loop and uses the legacy single-encoder path.

Command Reference (selected)
----------------------------
- Core IO and model
  - `--input`, `--out-prefix`, `--bands`, `--weights`, `--zero-shot-imagenet`, `--arch`, `--encoder`, `--seed`.
- Tiling and normalization
  - `--tile`, `--overlap`, `--feather/--no-feather`, `--global-norm`, `--norm-sample-tiles`, `--half`.
- Thresholding and masking
  - `--th` (DL), `--mask-talls`.
- Classic
  - `--classic/--no-classic` (default: classic ON), `--classic-modes`, `--classic-th`, `--classic-save-intermediate`.
- Fusion
  - `--fuse`, `--alpha`.
- Vectorization
  - `--vectorize`, `--min-area`, `--simplify`.
- Logging
  - `-v/--verbose` (repeat for DEBUG).

Workflow Details
----------------
- DL path
  - For each tile: read RGB/DSM/DTM, derive RVT layers, stack 9 channels, normalize, run U‑Net, sigmoid to probability.
  - Accumulate with feather weights and valid‑pixel masks; apply tall‑object filtering if set.
  - Threshold with `--th` and write GeoTIFFs.
- Classic path
  - For each tile: read DTM and compute per‑mode scores (rvtlog/hessian/morph), normalize to [0,1], accumulate with feathering.
  - If multiple modes: combine by mean of normalized probabilities; otherwise single mode is both the per‑mode and combined classic.
  - Threshold by Otsu or `--classic-th`, write per‑mode (optional) and combined GeoTIFFs.
- Fusion (optional)
  - α‑blend DL and classic probability maps; clamp to [0,1]; threshold with `--th` or default 0.5; write GeoTIFFs.
- Vectorization (optional)
  - Connected components, `--min-area` filter, optional `--simplify`, attribute `score_mean` from probability map.

Implementation Notes
--------------------
- Code lives in `archaeo_detect.py`.
- RVT compatibility: the code handles both `rvt-py` and newer `rvt` API variants by probing function signatures and result shapes.
- Nodata handling: all intermediate arrays propagate NaNs; valid masks gate accumulation; outputs set prob nodata=NaN, mask nodata=0.
- CRS and area: geographic CRSs are re‑projected to EPSG:6933 for area calculation and mapped back where needed.

Performance and Memory
----------------------
- Prefer GPU with CUDA; enable mixed precision with `--half` (CUDA only).
- Increase `--tile` to reduce overhead; ensure available RAM/VRAM.
- Disable feathering (`--no-feather`) for speed if seams are acceptable.
- Use `--global-norm` for consistent appearance across large mosaics.

Troubleshooting
---------------
- No classic rasters
  - Ensure DTM band index > 0 in `--bands` (5th value). Classic defaults to ON; use `--no-classic` to disable.
- RVT import error
  - On Python 3.10: `pip install rvt-py`. On newer Python: `pip install rvt` (if wheels available). Otherwise switch to 3.10.
- Fiona/GeoPandas fail to install on Windows
  - Use `conda install geopandas fiona` or skip vectorization (`--vectorize` off). DL/classic rasters still produce.
- Empty vector output
  - Threshold too high or features too small; reduce `--th`/`--classic-th` or `--min-area`.
- Artifacts at tile edges
  - Ensure `--overlap` is large enough (e.g., 10–25% of `--tile`). Keep `--feather` on.

Known Limitations
-----------------
- Relief derivatives depend on RVT implementation; different versions may vary slightly.
- Zero‑shot DL is exploratory; trained weights yield better results.
- Vectorization assumes relatively clean masks; very noisy probabilities may fragment polygons.

Reproducibility
---------------
- Uses fixed random seeds for NumPy and Torch (`--seed`). CUDA runs with deterministic algorithms where possible.

Acknowledgements and License
----------------------------
- Built on NumPy, SciPy, Rasterio, RVT, PyTorch, and SMP.
- Copyright remains with the repository owner.
