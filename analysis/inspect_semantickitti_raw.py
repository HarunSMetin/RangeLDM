"""Inspect SemanticKITTI scans and the repo's range-image conversion.

Usage:
  export SEMANTICKITTI_DATASET=/path/to/SemanticKITTI   # root or .../dataset/sequences
  python analysis/inspect_semantickitti_raw.py --max_files 3 --out_dir analysis/outputs/semantickitti

This script:
- Validates the expected SemanticKITTI structure (sequences/XX/velodyne/*.bin).
- Loads a few scans and prints xyz/intensity stats.
- Converts each scan to (H,W,2) range image with the repo's converter,
  applies missing-value fill + normalization, and saves quicklook PNGs.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np


def _iter_velodyne_bins(sk_root: Path) -> Iterable[Path]:
    # Accept either .../dataset/sequences or .../sequences or dataset root
    if (sk_root / "dataset" / "sequences").is_dir():
        base = sk_root / "dataset" / "sequences"
    elif sk_root.name == "sequences":
        base = sk_root
    else:
        base = sk_root / "sequences"
    for seq in sorted([d for d in base.iterdir() if d.is_dir()]):
        vel = seq / "velodyne"
        if vel.is_dir():
            yield from sorted(vel.glob("*.bin"))


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
        "--semantickitti_root",
        type=str,
        default=os.environ.get("SEMANTICKITTI_DATASET"),
        help="Path to SemanticKITTI root (or sequences folder).",
    )
    parser.add_argument("--max_files", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="analysis/outputs/semantickitti")
    args = parser.parse_args()

    if not args.semantickitti_root:
        raise SystemExit("Missing SemanticKITTI path. Provide --semantickitti_root or set SEMANTICKITTI_DATASET.")

    sk_root = Path(args.semantickitti_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bins = list(_iter_velodyne_bins(sk_root))
    if not bins:
        raise SystemExit(f"No velodyne .bin files found under {sk_root} (expected sequences/XX/velodyne/*.bin)")

    import sys
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from ldm.semantickitti_range_image import point_cloud_to_range_image_SemKITTI  # type: ignore

    to_range = point_cloud_to_range_image_SemKITTI(width=1024)

    for i, bin_path in enumerate(bins[: args.max_files]):
        pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        xyz = pts[:, :3]
        intensity = pts[:, 3]
        depth = np.linalg.norm(xyz, axis=1)

        print("\n===", bin_path, "===")
        print(f"points: {pts.shape[0]}")
        print(f"xyz min: {xyz.min(axis=0)}")
        print(f"xyz max: {xyz.max(axis=0)}")
        print(f"depth min/max/mean: {depth.min():.3f} / {depth.max():.3f} / {depth.mean():.3f}")
        print(f"intensity min/max/mean: {intensity.min():.3f} / {intensity.max():.3f} / {intensity.mean():.3f}")

        range_img = to_range(pts.copy())  # (H,W,2) with -1 missing
        range_img, mask, _ = to_range.process_miss_value(range_img)
        range_img_norm = to_range.normalize(range_img.copy())

        _save_grayscale_png(range_img[:, :, 0], out_dir / f"{i:02d}_range_raw.png")
        _save_grayscale_png(range_img[:, :, 1], out_dir / f"{i:02d}_intensity_raw.png")
        _save_grayscale_png(range_img_norm[:, :, 0], out_dir / f"{i:02d}_range_norm.png")

        missing_ratio = float((mask == 0).mean())
        print(f"range image shape (H,W,C): {range_img.shape}")
        print(f"missing ratio (before fill, approx): {missing_ratio:.3f}")

    print(f"\nSaved quicklook images to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
