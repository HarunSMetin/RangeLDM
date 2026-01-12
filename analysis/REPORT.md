# RangeLDM repo: training → evaluation → inference (teaching-style report)

This report explains what happens in this repository end-to-end, focusing on:
- Which datasets it actually uses (KITTI-360 / nuScenes)
- Which *parts* of those datasets are used
- How raw LiDAR data is converted into tensors (dimensions, channels)
- What “VAE” vs “LDM/DM” training does
- What inference scripts generate and how outputs are saved

---

## 0) Big picture

This repo generates **realistic LiDAR point clouds** by working in a **range-image representation**.

Pipeline (conceptual):
1) Take raw LiDAR point cloud (e.g., KITTI-360 Velodyne `.bin`).
2) Convert it into a **range image** (a 2D grid indexed by laser row + azimuth angle).
3) Train either:
   - **RangeDM** (diffusion directly in range-image space), or
   - **RangeLDM** (a latent diffusion model, where a **VAE** compresses the range image to a smaller latent, and diffusion happens in the latent).
4) At inference time, sample a synthetic range image / latent, then convert back to a point cloud and save `.bin`.

Key repo folders:
- `vae/`: trains the VAE (autoencoder) on range images
- `ldm/`: trains diffusion models and runs inference
- `metrics/`: evaluation code

---

## 1) Which dataset(s) are used?

### KITTI-360 (this repo’s usage)
Even though KITTI-360 provides many modalities (images, semantics, etc.), **this repo’s core pipeline uses only 3D LiDAR Velodyne scans**.

Used files:
- `data_3d_raw/*/velodyne_points/data/*.bin`

Not used by default:
- Fisheye/perspective camera images
- 2D semantic labels
- 3D bounding boxes
- SICK scans

#### Train/test split used in code
Defined in [ldm/kitti360_range_image.py](../ldm/kitti360_range_image.py):
- Train = all sequences **except** `0000_sync` and `0002_sync`
- Test/val = sequences **only** `0000_sync` and `0002_sync`

So when you download KITTI-360, the *minimum* you need for training is the **Velodyne raw scans** for all drives.

### nuScenes (this repo’s usage)
Used files:
- Metadata JSON:
  - `v1.0-trainval/sample_data.json` (train)
  - `v1.0-test/sample_data.json` (test)
- LiDAR sweeps referenced by that JSON (filenames containing `sweeps/LIDAR_TOP`)

Each nuScenes sweep is loaded as float32 and reshaped to `(-1, 5)`:
- `x,y,z,intensity,ring`

Implementation: [ldm/nuscenes_range_image.py](../ldm/nuscenes_range_image.py)

---

## 2) Raw data → Range image (what is the tensor shape?)

The *central data object* for both training and inference is the range image.

### Range image definition
A range image is a 2D grid with:
- One axis = azimuth angle (sweeping around 360°, treated as circular)
- One axis = laser ring/vertical channel (e.g., 64 rings for KITTI-360 HDL-64)

In this repo, a range image is constructed as a numpy array of shape:
- `(H, W, 2)` where:
  - `H = number of rings` (KITTI-360: 64, nuScenes: 32)
  - `W = horizontal resolution` (default 1024)
  - channel 0 = range/depth-like value
  - channel 1 = remission/intensity

Then it is converted to torch as:
- `torch.Tensor` with shape `(C, W, H)`
  - yes: **(C, W, H)** (not the common (C,H,W))
  - everywhere else in the repo they treat tensors as `(B, C, W, H)`

This happens in [ldm/dataset.py](../ldm/dataset.py) inside `RangeDataset.__getitem__`.

### Missing values
During projection, some pixels have no point; they are `-1`.
Then the repo fills missing values in two steps:
1) Fill with a 1-pixel horizontal shift (wrap-around)
2) Remaining missing pixels become a fixed “far range” value (`range_fill_value = [100, 0]`)

Code: `point_cloud_to_range_image.process_miss_value()` in [ldm/dataset.py](../ldm/dataset.py)

### Normalization
For KITTI-360 (default):
- `range = (range - mean) / std` with `mean=20`, `std=40`

For nuScenes, configs override:
- `mean=50`, `std=50`

This normalization only affects the range channel (channel 0) and is skipped for log/inverse modes.

---

## 3) Training pipeline

### 3.1 VAE training (`vae/`)
Entrypoint in README:
- `cd vae`
- `python main.py --base configs/kitti360.yaml`

What happens:
- Uses the `sgm` framework in `vae/sgm/`.
- Trains an autoencoder on range images with:
  - input channels: 2 (range + intensity)
  - latent channels: 4 (`z_channels: 4`)
  - effective input resolution: 256 in the VAE config (it downsamples internally)

Config: [vae/configs/kitti360.yaml](../vae/configs/kitti360.yaml)
- Dataset loader target: `sgm.data.kitti360_range_image.KITTIRangeLoader`
- Loss includes LPIPS discriminator but perceptual weight is set to 0.0 in this config.

Output:
- A checkpoint that is later converted/loaded into Diffusers `AutoencoderKL` format by `ldm/convert_vae.py`.

### 3.2 Diffusion training (`ldm/`)

There are 2 main training scripts:

#### (A) Unconditional generation
- [ldm/train_unconditional.py](../ldm/train_unconditional.py)
- Typical command:
  - `accelerate launch train_unconditional.py --cfg configs/RangeLDM.yaml`

This learns a model that samples from noise → synthetic range image (or latent).
It supports:
- **RangeDM** (`with_vae: False`) → diffusion directly outputs a 2-channel range image
- **RangeLDM** (`with_vae: True`) → diffusion happens in VAE latent space (typically 4 channels)

#### (B) Conditional generation (upsampling or inpainting)
- [ldm/train_conditional.py](../ldm/train_conditional.py)
- Typical command:
  - `accelerate launch train_conditional.py --cfg configs/upsample.yaml`

Two conditional modes exist (chosen by config):
1) **Upsampling / densification** (`upsample: 4`)
   - Input condition: a downsampled range image stored as `batch["down"]`
   - Condition encoder: `SparseRangeImageEncoder2` packs 4 neighboring W-pixels → channels
   - Condition channels: `2 * 4 = 8`
   - UNet input = noisy latent (4) + condition (8) → 12 channels

2) **Inpainting** (`inpainting: 0.0625` etc.)
   - Input condition: `masked_image` plus `inpainting_mask`
   - The masked image is encoded to latent (4 channels) and concatenated with the 1-channel mask
   - UNet input = noisy latent (4) + (masked_latent 4 + mask 1) → 9 channels

---

## 4) Inference (sampling) pipeline

### 4.1 Unconditional inference
Script: [ldm/inference.py](../ldm/inference.py)

How it chooses dataset:
- If config has `nuscenes: True` → uses `NUSCENES_DATASET` env var
- Else → uses `KITTI360_DATASET` env var

Outputs:
- Generates range images (or latents) using a Diffusers pipeline
- Converts generated output back to point cloud using `to_range.to_pc_torch(...)`
- Saves:
  - `generated/<idx>.bin` (point cloud)
  - `generated/<idx>.png` (BEV occupancy-like visualization)
  - `generated/<idx>_range.png` (range-image visualization)

### 4.2 Conditional inference (densification / inpainting)
Script: [ldm/inference_conditional.py](../ldm/inference_conditional.py)

- Loads a batch from the dataset test loader
- Runs the conditional pipeline
- Writes three folders:
  - input / target / result point clouds + BEV images

---

## 5) Why “circular” convolutions?

The azimuth dimension wraps around 360°. The repo treats that axis as circular by padding with `mode="circular"` before convolution.

Implementation: [ldm/utils.py](../ldm/utils.py)
- `Conv2d(... circular=True)`
- `replace_conv()` replaces standard convs with circular versions

This is a key architectural choice for range images.

---

## 6) Where does “testing / evaluation” happen?

Evaluation is under `metrics/`.
Start here: [metrics/metrics.md](../metrics/metrics.md)

Typical flow:
1) Generate point clouds with inference
2) Run metrics scripts comparing generated vs real distributions

---

## 7) Your KITTI-360 download question (which parts do you need?)

From the KITTI-360 download page you posted, the **minimum** for this repo is:
- **3D data & labels → Raw Velodyne Scans** (the `.bin` scans)

The repo does not need the 2D camera images/labels unless you plan to add your own conditioning.

---

## 8) “Analyze raw data” tools included in this repo

See the new analysis folder:
- [analysis/README.md](README.md)
- [analysis/inspect_kitti360_raw.py](inspect_kitti360_raw.py)
- [analysis/inspect_nuscenes_raw.py](inspect_nuscenes_raw.py)

These scripts print raw point stats and save range-image quicklooks so you can verify your dataset is correctly read by the repo.
