"""Inspect raw KITTI-360 Velodyne scans and the repo's range-image conversion.

Usage (PowerShell):
  $env:KITTI360_DATASET = "E:/datasets/KITTI-360"
  python analysis/inspect_kitti360_raw.py --max_files 3 --out_dir analysis/outputs/kitti360

What this does:
- Verifies expected KITTI-360 directory structure used by this repo.
- Loads a few *.bin scans (x,y,z,intensity) and prints basic stats.
- Converts each scan to a (H,W,2) range image, applies missing-value filling + normalization,
  and saves quicklook PNGs for range and intensity.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np


def _iter_velodyne_bins(kitti_root: Path) -> Iterable[Path]:
    yield from sorted(kitti_root.glob("data_3d_raw/*/velodyne_points/data/*.bin"))


def _save_grayscale_png(array_hw: np.ndarray, out_path: Path) -> None:
    # Avoid hard dependency on matplotlib; use Pillow.
    from PIL import Image

    array_hw = np.asarray(array_hw)
    array_hw = np.nan_to_num(array_hw)
    mn = float(array_hw.min())
    mx = float(array_hw.max())
    if mx <= mn:
        img = np.zeros_like(array_hw, dtype=np.uint8)
    else:
        img = ((array_hw - mn) / (mx - mn) * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(img, mode="L").save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kitti_root",
        type=str,
        default=os.environ.get("KITTI360_DATASET"),
        help="Path to KITTI-360 root folder (or set KITTI360_DATASET env var).",
    )
    parser.add_argument("--max_files", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="analysis/outputs/kitti360")
    args = parser.parse_args()

    if not args.kitti_root:
        raise SystemExit("Missing KITTI-360 root path. Provide --kitti_root or set KITTI360_DATASET.")

    kitti_root = Path(args.kitti_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    expected = [
        kitti_root / "data_3d_raw",
        kitti_root / "calibration",
        kitti_root / "data_poses",
    ]
    missing = [p for p in expected if not p.exists()]
    if missing:
        print("Expected KITTI-360 folders not found:")
        for p in missing:
            print(f"  - {p}")
        print("\nThis repo only needs data_3d_raw/*/velodyne_points/data/*.bin for training.")

    bins = list(_iter_velodyne_bins(kitti_root))
    if not bins:
        raise SystemExit(f"No Velodyne .bin files found under: {kitti_root}/data_3d_raw/*/velodyne_points/data/*.bin")

    # Import the repo's converter lazily (keeps script runnable even if deps aren't installed).
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from ldm.kitti360_range_image import point_cloud_to_range_image_KITTI  # type: ignore

    to_range = point_cloud_to_range_image_KITTI(width=1024)

    for i, bin_path in enumerate(bins[: args.max_files]):
        pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        xyz = pts[:, :3]
        intensity = pts[:, 3]
        depth = np.linalg.norm(xyz, axis=1)

        print("\n===", bin_path.relative_to(kitti_root), "===")
        print(f"points: {pts.shape[0]}")
        print(f"xyz min: {xyz.min(axis=0)}")
        print(f"xyz max: {xyz.max(axis=0)}")
        print(f"depth min/max/mean: {depth.min():.3f} / {depth.max():.3f} / {depth.mean():.3f}")
        print(f"intensity min/max/mean: {intensity.min():.3f} / {intensity.max():.3f} / {intensity.mean():.3f}")

        range_img = to_range(pts.copy())  # (H,W,2) with -1 missing
        range_img, mask, _car_window = to_range.process_miss_value(range_img)
        range_img_norm = to_range.normalize(range_img.copy())

        # Quicklooks
        _save_grayscale_png(range_img[:, :, 0], out_dir / f"{i:02d}_range_raw.png")
        _save_grayscale_png(range_img[:, :, 1], out_dir / f"{i:02d}_intensity_raw.png")
        _save_grayscale_png(range_img_norm[:, :, 0], out_dir / f"{i:02d}_range_norm.png")

        missing_ratio = float((mask == 0).mean())
        print(f"range image shape (H,W,C): {range_img.shape} (H={range_img.shape[0]}, W={range_img.shape[1]}, C=2)")
        print(f"missing ratio (before fill, approx): {missing_ratio:.3f}")

    print(f"\nSaved quicklook images to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
