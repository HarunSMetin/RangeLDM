"""Inspect raw nuScenes LIDAR_TOP scans and the repo's range-image conversion.

Usage (PowerShell):
  $env:NUSCENES_DATASET = "E:/datasets/nuScenes"
  python analysis/inspect_nuscenes_raw.py --max_files 3 --out_dir analysis/outputs/nuscenes

This repo reads nuScenes sweeps as:
  <NUSCENES_ROOT>/v1.0-trainval/sample_data.json (or v1.0-test/sample_data.json)
  and then loads files whose filename contains 'sweeps/LIDAR_TOP'.

Each loaded scan is float32 with shape (-1, 5): x,y,z,intensity,ring.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def _save_grayscale_png(array_hw: np.ndarray, out_path: Path) -> None:
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
        "--nuscenes_root",
        type=str,
        default=os.environ.get("NUSCENES_DATASET"),
        help="Path to nuScenes root folder (or set NUSCENES_DATASET env var).",
    )
    parser.add_argument("--split", type=str, default="trainval", choices=["trainval", "test"])
    parser.add_argument("--max_files", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="analysis/outputs/nuscenes")
    args = parser.parse_args()

    if not args.nuscenes_root:
        raise SystemExit("Missing nuScenes root path. Provide --nuscenes_root or set NUSCENES_DATASET.")

    root = Path(args.nuscenes_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = root / ("v1.0-trainval" if args.split == "trainval" else "v1.0-test") / "sample_data.json"
    if not meta.exists():
        raise SystemExit(f"Missing metadata file: {meta}")

    with meta.open("r", encoding="utf-8") as f:
        sample_data = json.load(f)

    lidar_files = [root / x["filename"] for x in sample_data if "sweeps/LIDAR_TOP" in x.get("filename", "")]
    lidar_files = sorted(lidar_files)
    if not lidar_files:
        raise SystemExit("No 'sweeps/LIDAR_TOP' entries found in sample_data.json")

    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from ldm.nuscenes_range_image import point_cloud_to_range_image_nuScenes  # type: ignore

    to_range = point_cloud_to_range_image_nuScenes(width=1024)
    to_range.mean = 50.0
    to_range.std = 50.0

    for i, bin_path in enumerate(lidar_files[: args.max_files]):
        pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)
        xyz = pts[:, :3]
        intensity = pts[:, 3]
        ring = pts[:, 4]
        depth = np.linalg.norm(xyz, axis=1)

        print("\n===", bin_path.relative_to(root), "===")
        print(f"points: {pts.shape[0]}")
        print(f"xyz min: {xyz.min(axis=0)}")
        print(f"xyz max: {xyz.max(axis=0)}")
        print(f"depth min/max/mean: {depth.min():.3f} / {depth.max():.3f} / {depth.mean():.3f}")
        print(f"intensity min/max/mean: {intensity.min():.3f} / {intensity.max():.3f} / {intensity.mean():.3f}")
        print(f"ring min/max: {ring.min():.0f} / {ring.max():.0f}")

        range_img = to_range(pts.copy())
        range_img, mask, _ = to_range.process_miss_value(range_img)
        range_img_norm = to_range.normalize(range_img.copy())

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
