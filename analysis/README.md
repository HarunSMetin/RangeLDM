# Analysis folder

This folder contains small, runnable scripts to inspect the **raw dataset files** that RangeLDM actually uses.

Important: although KITTI-360 contains huge 2D camera data, this repo’s core training pipeline uses **3D LiDAR scans** (Velodyne `.bin`) and converts them into **range images**.

## 1) KITTI-360 raw inspection

PowerShell:

```powershell
$env:KITTI360_DATASET = "E:/datasets/KITTI-360"
python analysis/inspect_kitti360_raw.py --max_files 3 --out_dir analysis/outputs/kitti360
```

What you should see:
- Printed point statistics for a few scans
- Saved PNGs in `analysis/outputs/kitti360/`

## 2) nuScenes raw inspection

PowerShell:

```powershell
$env:NUSCENES_DATASET = "E:/datasets/nuScenes"
python analysis/inspect_nuscenes_raw.py --split trainval --max_files 3 --out_dir analysis/outputs/nuscenes
```

What you should see:
- Printed point statistics for a few sweeps
- Saved PNGs in `analysis/outputs/nuscenes/`

## Notes

- These scripts import conversion utilities from the repo (they add the repo root to `sys.path`).
- They do **not** require you to run training first.
- If you want deeper visualization (matplotlib, point cloud viewers), tell me what you prefer and I can extend this folder.
