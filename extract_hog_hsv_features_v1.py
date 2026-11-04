#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

try:
    import sklearn
    import skimage
    from skimage.feature import hog
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install with:\n"
        "  pip install scikit-image scikit-learn joblib\n"
        f"Original error: {exc}"
    )

EXPECTED_MANIFEST_PACKAGE_FP = "fe54fab7182f2aa5d35c987558f3deddd35e684ad40377a074efc6d92b4a19c6"
EXPECTED_ROI_DATASET_FP = "d25132b9afae3bc858d3470b0a6cc5253abf5085c7ba3d1596a6f4bdc3d7ce77"
FEATURE_VERSION = "HOG_HSV_V1"
FEATURE_DIM = 1812
CANVAS_SIZE = 64
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"
HSV_BINS = (16, 16, 16)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_single_hash(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip().split()[0]


def ensure_clean(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {path}. Use --overwrite-output.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def read_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        low_memory=False,
        dtype={
            "roi_id": "string",
            "crop_relpath": "string",
            "source_sha256": "string",
            "source_image_relpath": "string",
            "crop_pixel_sha256": "string",
            "assigned_class_name": "string",
            "label_status": "string",
        },
    )
    if "assigned_class_id" not in df.columns:
        raise RuntimeError(f"Missing assigned_class_id: {path}")
    df["assigned_class_id"] = pd.to_numeric(df["assigned_class_id"], errors="raise").astype(int)
    required = ["roi_id", "crop_relpath", "crop_pixel_sha256", "assigned_class_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in {path}: {missing}")
    return df


def resolve_crop_path(roi_root: Path, rel: str) -> Path:
    p = Path(str(rel))
    if p.is_absolute():
        return p
    candidate = roi_root / p
    if candidate.is_file():
        return candidate
    # tolerate manifests that accidentally include the root directory name
    parts = p.parts
    if parts and parts[0] == roi_root.name:
        candidate2 = roi_root.parent / p
        if candidate2.is_file():
            return candidate2
    return candidate


def resize_preserve_aspect(image: np.ndarray, canvas: int = CANVAS_SIZE):
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Invalid crop dimensions")
    scale = min(canvas / float(w), canvas / float(h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (nw, nh), interpolation=interp)

    top = (canvas - nh) // 2
    bottom = canvas - nh - top
    left = (canvas - nw) // 2
    right = canvas - nw - left

    # Reflective padding avoids a strong artificial black frame in HOG.
    # For 1-pixel dimensions REFLECT_101 can fail, so fall back to REPLICATE.
    border = cv2.BORDER_REFLECT_101 if min(nh, nw) > 1 else cv2.BORDER_REPLICATE
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, border)
    if padded.shape[:2] != (canvas, canvas):
        raise RuntimeError(f"Letterbox shape mismatch: {padded.shape}")
    return resized, padded


def hsv_histogram(content_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(content_bgr, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [HSV_BINS[0]], [0, 180]).reshape(-1)
    hist_s = cv2.calcHist([hsv], [1], None, [HSV_BINS[1]], [0, 256]).reshape(-1)
    hist_v = cv2.calcHist([hsv], [2], None, [HSV_BINS[2]], [0, 256]).reshape(-1)
    parts = []
    for hist in (hist_h, hist_s, hist_v):
        denom = float(hist.sum())
        if denom > 0:
            hist = hist / denom
        parts.append(hist.astype(np.float32, copy=False))
    return np.concatenate(parts).astype(np.float32, copy=False)


def extract_feature(image_bgr: np.ndarray) -> np.ndarray:
    content, canvas = resize_preserve_aspect(image_bgr, CANVAS_SIZE)
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    hog_vec = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        visualize=False,
        feature_vector=True,
        transform_sqrt=False,
    ).astype(np.float32, copy=False)
    color_vec = hsv_histogram(content)
    feat = np.concatenate([hog_vec, color_vec]).astype(np.float32, copy=False)
    if feat.shape != (FEATURE_DIM,):
        raise RuntimeError(f"Feature dimension {feat.shape} != ({FEATURE_DIM},)")
    if not np.isfinite(feat).all():
        raise RuntimeError("Non-finite feature detected")
    return feat


def main():
    ap = argparse.ArgumentParser(description="Extract canonical HOG+HSV features for VALID-only SVM/RF selection.")
    ap.add_argument("--roi-root", type=Path, default=Path("ROI-Dataset-Frozen-v1"))
    ap.add_argument("--manifests-root", type=Path, default=Path("classifier_manifests_v1"))
    ap.add_argument("--outdir", type=Path, default=Path("hog_hsv_features_v1"))
    ap.add_argument("--overwrite-output", action="store_true")
    ap.add_argument("--print-every", type=int, default=1000)
    args = ap.parse_args()

    manifest_fp = read_single_hash(args.manifests_root / "classifier_manifest_package_fingerprint.sha256")
    if manifest_fp != EXPECTED_MANIFEST_PACKAGE_FP:
        raise RuntimeError(f"Classifier manifest fingerprint mismatch: {manifest_fp}")
    roi_fp = read_single_hash(args.roi_root / "roi_dataset_fingerprint.sha256")
    if roi_fp != EXPECTED_ROI_DATASET_FP:
        raise RuntimeError(f"ROI dataset fingerprint mismatch: {roi_fp}")

    # No TEST manifest is read by this extractor.
    bg2 = read_manifest(args.manifests_root / "bg2x" / "train.csv")
    valid = read_manifest(args.manifests_root / "shared" / "valid_natural.csv")
    bg1 = read_manifest(args.manifests_root / "bg1x" / "train.csv")

    bg1_ids = set(bg1["roi_id"].astype(str))
    bg2_ids = set(bg2["roi_id"].astype(str))
    if not bg1_ids.issubset(bg2_ids):
        raise RuntimeError("BG1x train is not a subset of BG2x train")

    union = pd.concat([bg2, valid], ignore_index=True)
    if union["roi_id"].duplicated().any():
        dup = int(union["roi_id"].duplicated().sum())
        raise RuntimeError(f"Unexpected duplicate roi_id across BG2x TRAIN and VALID: {dup}")
    union = union.sort_values("roi_id", kind="mergesort").reset_index(drop=True)

    ensure_clean(args.outdir, args.overwrite_output)
    n = len(union)
    feat_path = args.outdir / "features.float32.npy"
    features = np.lib.format.open_memmap(feat_path, mode="w+", dtype=np.float32, shape=(n, FEATURE_DIM))

    index_rows = []
    t0 = time.time()
    for i, row in union.iterrows():
        crop = resolve_crop_path(args.roi_root, str(row["crop_relpath"]))
        image = cv2.imread(str(crop), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read crop: {crop}")
        features[i] = extract_feature(image)
        index_rows.append({
            "feature_row": i,
            "roi_id": str(row["roi_id"]),
            "assigned_class_id": int(row["assigned_class_id"]),
            "assigned_class_name": str(row.get("assigned_class_name", "")),
            "crop_relpath": str(row["crop_relpath"]),
            "crop_pixel_sha256": str(row["crop_pixel_sha256"]),
            "set_membership": "train_bg2x" if str(row["roi_id"]) in bg2_ids else "valid_natural",
        })
        if (i + 1) % args.print_every == 0 or i + 1 == n:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            print(f"[{i+1:6d}/{n}] {rate:7.1f} crops/s | elapsed={elapsed/60:.1f} min", flush=True)

    features.flush()
    del features

    index = pd.DataFrame(index_rows)
    index_path = args.outdir / "feature_index.csv"
    index.to_csv(index_path, index=False, lineterminator="\n")

    config = {
        "feature_version": FEATURE_VERSION,
        "test_accessed": False,
        "parent_roi_dataset_fingerprint": roi_fp,
        "classifier_manifest_package_fingerprint": manifest_fp,
        "n_rows": n,
        "feature_dim": FEATURE_DIM,
        "canvas": [CANVAS_SIZE, CANVAS_SIZE],
        "aspect_ratio": "preserved",
        "padding": "BORDER_REFLECT_101; BORDER_REPLICATE fallback for 1-pixel dimension",
        "hog": {
            "grayscale": True,
            "orientations": HOG_ORIENTATIONS,
            "pixels_per_cell": list(HOG_PIXELS_PER_CELL),
            "cells_per_block": list(HOG_CELLS_PER_BLOCK),
            "block_norm": HOG_BLOCK_NORM,
            "transform_sqrt": False,
            "dimension": 1764,
        },
        "hsv": {
            "computed_on": "resized content before padding",
            "bins": {"H": 16, "S": 16, "V": 16},
            "ranges": {"H": [0, 180], "S": [0, 256], "V": [0, 256]},
            "normalization": "each 1D channel histogram sums to 1 when nonempty",
            "dimension": 48,
        },
        "versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_image": skimage.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    config_path = args.outdir / "feature_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    hashes = {
        "features.float32.npy": sha256_file(feat_path),
        "feature_index.csv": sha256_file(index_path),
        "feature_config.json": sha256_file(config_path),
    }
    package_basis = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    package_fp = hashlib.sha256(package_basis).hexdigest()
    (args.outdir / "file_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.outdir / "feature_cache_fingerprint.sha256").write_text(package_fp + "\n", encoding="utf-8")

    print("=" * 88)
    print("HOG+HSV FEATURE EXTRACTION COMPLETED")
    print("=" * 88)
    print(f"rows                : {n}")
    print(f"feature_dim         : {FEATURE_DIM}")
    print(f"BG2x train rows     : {len(bg2)}")
    print(f"VALID natural rows  : {len(valid)}")
    print("TEST accessed       : NO")
    print(f"feature cache fp    : {package_fp}")
    print(f"output              : {args.outdir}")
    print("=" * 88)


if __name__ == "__main__":
    main()
