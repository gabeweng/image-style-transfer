"""
End-to-end preprocessing pipeline for the Penn campus image dataset.

This script mirrors notebooks/00_preprocess_and_align.ipynb for cluster or
local runs: raw image conversion, manifest writing, LightGlue alignment,
shared valid cropping, duplicate filtering, and Hugging Face ImageFolder
export.

Example:
    python scripts/preprocess.py \
        --base /path/to/CIS_5190_group_project \
        --redo-all
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener
from tqdm.auto import tqdm


register_heif_opener()

MANIFEST_COLUMNS = [
    "file_name",
    "location",
    "time_of_day",
    "weather",
    "caption",
    "source_file",
    "original_file_name",
    "crop_w",
    "crop_h",
    "matches",
    "inliers",
    "inlier_ratio",
    "representative_score",
    "split",
    "status",
    "drop_reason",
]

VALID_IMAGE_SUFFIXES = {".heic", ".heif", ".jpg", ".jpeg", ".png"}
FILENAME_PATTERN = re.compile(r"^([^_]+)_([^_]+)_([a-zA-Z]+)")
DAYLIKE = {"daytime", "day", "morning"}


@dataclass
class PreprocessConfig:
    base: str
    raw_dir: str
    processed_dir: str
    aligned_dir: str
    filtered_dir: str
    manifest_csv: str
    hf_dataset_dir: str
    redo_preprocess: bool = False
    redo_alignment: bool = False
    redo_filtered: bool = False
    redo_hf_dataset: bool = False
    location_filter: set[str] | None = None
    output_size: int = 512
    jpeg_quality: int = 95
    resize: int = 1024
    max_keypoints: int = 2048
    ransac_reproj_threshold: float = 5.0
    min_matches: int = 12
    min_inliers: int = 8
    min_inlier_ratio: float = 0.0
    min_shared_crop_area_ratio: float = 0.50
    ecc_score_size: int = 768
    ecc_max_iters: int = 80
    ecc_eps: float = 1e-5
    ecc_max_worse_factor: float = 1.05
    ecc_max_corner_drift_frac: float = 0.20
    ecc_min_scale: float = 0.70
    ecc_max_scale: float = 1.30


cfg: PreprocessConfig
torch = None
device = None
extractor = None
matcher = None
load_image = None
alignment_failures: list[dict] = []
crop_drops: list[dict] = []


def import_lightglue():
    try:
        from lightglue import LightGlue, SuperPoint
        from lightglue.utils import load_image as lightglue_load_image
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise SystemExit(
            "Missing LightGlue. Install it with:\n"
            "  pip install git+https://github.com/cvg/LightGlue.git"
        ) from exc
    return LightGlue, SuperPoint, lightglue_load_image


def import_torch():
    try:
        import torch as torch_module
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise SystemExit(
            "Missing torch. Install project dependencies first:\n"
            "  pip install -r requirements.txt"
        ) from exc
    return torch_module


def clean_stem(value) -> str:
    stem = Path(str(value)).stem.lower().strip()
    stem = re.sub(r"\s+", "_", stem)
    stem = re.sub(r"[^a-z0-9_()\-]+", "_", stem)
    return stem.strip("_") or "image"


def clean_label(value) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def caption_from_labels(location, time_of_day, weather) -> str:
    loc = clean_label(location)
    tod = clean_label(time_of_day)
    wx = clean_label(weather)
    return f"A photo of {loc} at Penn during {tod} {wx} weather."


def split_for_location(location) -> str:
    return "val" if sum(ord(ch) for ch in str(location).lower()) % 5 == 0 else "train"


def ensure_empty_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def file_exists_and_nonempty(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def manifest_record_from_row(
    row,
    status: str,
    drop_reason: str = "",
    file_name: str | None = None,
    stats: dict | None = None,
    crop_w="",
    crop_h="",
    representative_score="",
) -> dict:
    stats = stats or {}
    row_get = row.get if hasattr(row, "get") else dict(row).get
    raw_ratio = stats.get("inlier_ratio", row_get("inlier_ratio", ""))
    inlier_ratio = ""
    if raw_ratio != "":
        inlier_ratio = round(float(raw_ratio or 0), 4)

    return {
        "file_name": file_name if file_name is not None else row_get("file_name", row_get("source_file", "")),
        "location": row_get("location", ""),
        "time_of_day": row_get("time_of_day", ""),
        "weather": row_get("weather", ""),
        "caption": row_get(
            "caption",
            caption_from_labels(row_get("location", ""), row_get("time_of_day", ""), row_get("weather", "")),
        ),
        "source_file": row_get("source_file", row_get("file_name", "")),
        "original_file_name": row_get("original_file_name", ""),
        "crop_w": crop_w if crop_w != "" else row_get("crop_w", ""),
        "crop_h": crop_h if crop_h != "" else row_get("crop_h", ""),
        "matches": stats.get("matches", row_get("matches", "")),
        "inliers": stats.get("inliers", row_get("inliers", "")),
        "inlier_ratio": inlier_ratio,
        "representative_score": representative_score
        if representative_score != ""
        else row_get("representative_score", ""),
        "split": row_get("split", split_for_location(row_get("location", ""))),
        "status": status,
        "drop_reason": drop_reason,
    }


def write_manifest(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in MANIFEST_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out = out[MANIFEST_COLUMNS]
    out.to_csv(cfg.manifest_csv, index=False)
    return out


def summarize_manifest(df: pd.DataFrame, title: str, path_base: str | None = None) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print("rows:", len(df))
    if "split" in df.columns and len(df):
        print("split:", df["split"].value_counts().to_dict())
    if "status" in df.columns and len(df):
        print("status:", df["status"].value_counts().to_dict())
    if {"time_of_day", "weather"}.issubset(df.columns) and len(df):
        print("time/weather:", df.groupby(["time_of_day", "weather"]).size().to_dict())
    if "location" in df.columns and len(df):
        print("top locations:", df["location"].value_counts().head(10).to_dict())
    if path_base is not None and "file_name" in df.columns and len(df):
        existing = sum(os.path.exists(os.path.join(path_base, str(p))) for p in df["file_name"])
        print("existing files:", existing, "/", len(df))


def parse_filename(fname: str):
    match = FILENAME_PATTERN.match(fname)
    if not match:
        return None
    location, tod, weather = match.groups()
    weather = "rainy" if weather.lower() == "rain" else weather.lower()
    return location.lower(), tod.lower(), weather, clean_stem(fname)


def preprocess_complete() -> bool:
    if not file_exists_and_nonempty(cfg.manifest_csv):
        return False
    try:
        df = pd.read_csv(cfg.manifest_csv)
    except Exception:
        return False
    required = {
        "file_name",
        "source_file",
        "original_file_name",
        "location",
        "time_of_day",
        "weather",
        "caption",
        "split",
    }
    if not required.issubset(df.columns) or df.empty:
        return False
    missing = [
        p for p in df["source_file"].astype(str) if not os.path.exists(os.path.join(cfg.processed_dir, p))
    ]
    if missing:
        print("Processed files missing from existing manifest:", missing[:5])
        return False
    return True


def run_preprocess() -> pd.DataFrame:
    assert os.path.isdir(cfg.raw_dir), f"Missing raw image directory: {cfg.raw_dir}"
    if cfg.redo_preprocess:
        print("redo_preprocess=True; deleting previous processed images and manifest.")
        ensure_empty_dir(cfg.processed_dir)
        if os.path.exists(cfg.manifest_csv):
            os.remove(cfg.manifest_csv)
    elif preprocess_complete():
        print("Preprocess already complete; skipping.")
        return pd.read_csv(cfg.manifest_csv)
    else:
        os.makedirs(cfg.processed_dir, exist_ok=True)

    parsed_files = []
    skipped_non_images = []
    badly_named = []
    skipped_unreadable = []

    for fname in sorted(os.listdir(cfg.raw_dir)):
        suffix = os.path.splitext(fname)[1].lower()
        if ":zone.identifier" in fname.lower() or suffix not in VALID_IMAGE_SUFFIXES:
            skipped_non_images.append(fname)
            continue
        parsed = parse_filename(fname)
        if parsed is None:
            badly_named.append(fname)
            continue
        parsed_files.append((fname, *parsed))

    location_to_index = {loc: idx for idx, loc in enumerate(sorted({row[1] for row in parsed_files}))}
    records = []

    for fname, location, tod, weather, stem in tqdm(parsed_files, desc="Preprocess images"):
        in_path = os.path.join(cfg.raw_dir, fname)
        location_index = location_to_index[location]
        condition_dir = f"{tod}_{weather}"
        out_name = f"{stem}.jpg"
        rel_path = os.path.join(str(location_index), condition_dir, out_name).lower()
        out_path = os.path.join(cfg.processed_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        try:
            img = ImageOps.exif_transpose(Image.open(in_path)).convert("RGB")
            img.save(out_path, "JPEG", quality=cfg.jpeg_quality)
        except (UnidentifiedImageError, OSError) as exc:
            skipped_unreadable.append((fname, str(exc)))
            continue

        records.append(
            {
                "file_name": rel_path,
                "location": location,
                "time_of_day": tod,
                "weather": weather,
                "caption": caption_from_labels(location, tod, weather),
                "source_file": rel_path,
                "original_file_name": fname.lower(),
                "crop_w": "",
                "crop_h": "",
                "matches": "",
                "inliers": "",
                "inlier_ratio": "",
                "representative_score": "",
                "split": split_for_location(location),
                "status": "processed",
                "drop_reason": "",
            }
        )

    df = pd.DataFrame(records).sort_values(["location", "time_of_day", "weather", "source_file"])
    df = write_manifest(df)
    print(f"Saved {len(df)} rows to {cfg.manifest_csv}")
    if badly_named:
        print(f"Badly named files ({len(badly_named)}):", badly_named[:20])
    if skipped_non_images:
        print(f"Skipped non-image sidecar/files ({len(skipped_non_images)}):", skipped_non_images[:20])
    if skipped_unreadable:
        print(f"Skipped unreadable image files ({len(skipped_unreadable)}):", skipped_unreadable[:10])
    return df


def alignment_complete() -> bool:
    if not file_exists_and_nonempty(cfg.manifest_csv):
        return False
    try:
        df = pd.read_csv(cfg.manifest_csv)
    except Exception:
        return False
    if df.empty or "file_name" not in df.columns:
        return False
    aligned_rows = df[df["file_name"].astype(str).str.startswith("aligned/")]
    if aligned_rows.empty:
        return False
    missing = [
        p for p in aligned_rows["file_name"].astype(str) if not os.path.exists(os.path.join(cfg.base, p))
    ]
    if missing:
        print("Aligned files missing from existing manifest:", missing[:5])
        return False
    return True


def pick_anchor(group: pd.DataFrame) -> pd.Series:
    tod = group["time_of_day"].astype(str).str.lower().str.strip()
    weather = group["weather"].astype(str).str.lower().str.strip()
    candidates = group[tod.isin(DAYLIKE) & (weather == "clear")]
    if candidates.empty:
        candidates = group[tod.isin(DAYLIKE)]
    if candidates.empty:
        candidates = group
    return candidates.iloc[0]


def load_bgr_checked(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def lightglue_match(anchor_path: str, target_path: str):
    image0_raw = load_image(anchor_path, resize=cfg.resize)
    image1_raw = load_image(target_path, resize=cfg.resize)
    image0 = image0_raw.mean(dim=0, keepdim=True).unsqueeze(0).to(device)
    image1 = image1_raw.mean(dim=0, keepdim=True).unsqueeze(0).to(device)

    with torch.inference_mode():
        feats0 = extractor({"image": image0})
        feats1 = extractor({"image": image1})
        matches01 = matcher({"image0": feats0, "image1": feats1})

    kpts0 = feats0["keypoints"][0].detach().cpu().numpy()
    kpts1 = feats1["keypoints"][0].detach().cpu().numpy()
    matches = matches01["matches"][0].detach().cpu().numpy()
    if len(matches) == 0:
        return None, {"matches": 0, "inliers": 0, "inlier_ratio": 0.0, "reason": "no_matches"}
    return (image0_raw, image1_raw, kpts0[matches[:, 0]], kpts1[matches[:, 1]], len(matches)), None


def estimate_target_to_anchor(anchor_bgr, target_bgr, anchor_path: str, target_path: str):
    match_result, error = lightglue_match(anchor_path, target_path)
    if error is not None:
        return None, error

    image0_raw, image1_raw, mkpts0, mkpts1, n_matches = match_result
    if n_matches < cfg.min_matches:
        return None, {"matches": n_matches, "inliers": 0, "inlier_ratio": 0.0, "reason": "not_enough_matches"}

    h0, w0 = anchor_bgr.shape[:2]
    h1, w1 = target_bgr.shape[:2]
    scale0 = np.array([w0 / image0_raw.shape[2], h0 / image0_raw.shape[1]])
    scale1 = np.array([w1 / image1_raw.shape[2], h1 / image1_raw.shape[1]])
    mkpts0_orig = mkpts0 * scale0
    mkpts1_orig = mkpts1 * scale1

    H, inlier_mask = cv2.findHomography(
        mkpts1_orig,
        mkpts0_orig,
        cv2.USAC_MAGSAC,
        cfg.ransac_reproj_threshold,
    )
    if H is None or inlier_mask is None:
        return None, {"matches": n_matches, "inliers": 0, "inlier_ratio": 0.0, "reason": "homography_failed"}

    inliers = int(inlier_mask.ravel().sum())
    inlier_ratio = inliers / max(n_matches, 1)
    if inliers < cfg.min_inliers:
        return None, {"matches": n_matches, "inliers": inliers, "inlier_ratio": inlier_ratio, "reason": "too_few_inliers"}
    if cfg.min_inlier_ratio > 0 and inlier_ratio < cfg.min_inlier_ratio:
        return None, {
            "matches": n_matches,
            "inliers": inliers,
            "inlier_ratio": inlier_ratio,
            "reason": "too_low_inlier_ratio",
        }

    return H, {"matches": n_matches, "inliers": inliers, "inlier_ratio": inlier_ratio, "reason": "ok"}


def resize_output(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (cfg.output_size, cfg.output_size), interpolation=cv2.INTER_AREA)


def unique_output_name(location_dir: str, stem: str) -> str:
    base = f"{stem}_aligned.jpg"
    candidate = base
    counter = 2
    while os.path.exists(os.path.join(location_dir, candidate)):
        candidate = f"{stem}_aligned_{counter}.jpg"
        counter += 1
    return candidate


def shared_crop_for_items(items: list[dict], anchor_area: int):
    shared_mask = items[0]["mask"]
    for item in items[1:]:
        shared_mask = cv2.bitwise_and(shared_mask, item["mask"])
    coords = cv2.findNonZero(shared_mask)
    if coords is None:
        return None, 0.0
    x, y, crop_w, crop_h = cv2.boundingRect(coords)
    return (x, y, crop_w, crop_h), (crop_w * crop_h) / max(1, anchor_area)


def prune_to_valid_shared_crop(items: list[dict], anchor_area: int):
    dropped = []
    while True:
        crop, area_ratio = shared_crop_for_items(items, anchor_area)
        if crop is not None and area_ratio >= cfg.min_shared_crop_area_ratio:
            return items, dropped, crop, area_ratio
        if len(items) <= 1:
            return items, dropped, crop, area_ratio

        best_idx = None
        best_area = -1.0
        for idx in range(1, len(items)):
            trial = items[:idx] + items[idx + 1 :]
            _, trial_area = shared_crop_for_items(trial, anchor_area)
            if trial_area > best_area:
                best_idx = idx
                best_area = trial_area
        dropped_item = items.pop(best_idx)
        dropped.append(dropped_item)
        crop_drops.append(
            manifest_record_from_row(
                dropped_item["row"],
                "crop_dropped",
                "excessive_crop_loss",
                stats=dropped_item["stats"],
            )
        )
        print(f"  [DROP] {dropped_item['row']['source_file']} caused excessive crop loss")


def align_location_group(location: str, group: pd.DataFrame) -> list[dict]:
    group = group.copy().reset_index(drop=True)
    anchor_row = pick_anchor(group)
    anchor_source = anchor_row["source_file"]
    anchor_path = os.path.join(cfg.processed_dir, anchor_source)

    try:
        anchor_bgr = load_bgr_checked(anchor_path)
    except FileNotFoundError:
        print(f"[SKIP] {location}: missing anchor {anchor_path}")
        return []

    h0, w0 = anchor_bgr.shape[:2]
    anchor_area = h0 * w0
    aligned_items = [
        {
            "row": anchor_row,
            "image": anchor_bgr,
            "mask": np.ones((h0, w0), dtype=np.uint8) * 255,
            "stats": {"matches": 0, "inliers": 0, "inlier_ratio": 1.0, "reason": "anchor"},
            "is_anchor": True,
        }
    ]

    print(f"\nProcessing {location}: {len(group)} images | anchor={anchor_source}")
    for _, row in group.iterrows():
        if row["source_file"] == anchor_source:
            continue

        target_source = row["source_file"]
        target_path = os.path.join(cfg.processed_dir, target_source)
        try:
            target_bgr = load_bgr_checked(target_path)
        except FileNotFoundError:
            alignment_failures.append(
                manifest_record_from_row(
                    row,
                    "alignment_failed",
                    "missing_file",
                    stats={"matches": 0, "inliers": 0, "inlier_ratio": 0.0},
                )
            )
            print(f"  [MISS] {target_source}")
            continue

        H, stats = estimate_target_to_anchor(anchor_bgr, target_bgr, anchor_path, target_path)
        if H is None:
            alignment_failures.append(manifest_record_from_row(row, "alignment_failed", stats["reason"], stats=stats))
            print(
                f"  [FAIL] {target_source} matches={stats['matches']} "
                f"inliers={stats['inliers']} ratio={stats['inlier_ratio']:.2f} "
                f"reason={stats['reason']}"
            )
            continue

        aligned = cv2.warpPerspective(target_bgr, H, (w0, h0))
        source_mask = np.ones(target_bgr.shape[:2], dtype=np.uint8) * 255
        warped_mask = cv2.warpPerspective(source_mask, H, (w0, h0))
        aligned_items.append(
            {
                "row": row,
                "image": aligned,
                "mask": warped_mask,
                "stats": stats,
                "is_anchor": False,
            }
        )
        print(f"  [OK] {target_source} matches={stats['matches']} inliers={stats['inliers']} ratio={stats['inlier_ratio']:.2f}")

    aligned_items, dropped, crop, area_ratio = prune_to_valid_shared_crop(aligned_items, anchor_area)
    if crop is None:
        print(f"  [SKIP] {location}: no shared crop after pruning")
        return []

    x, y, crop_w, crop_h = crop
    location_dir = os.path.join(cfg.aligned_dir, clean_stem(location))
    os.makedirs(location_dir, exist_ok=True)

    records = []
    for item in aligned_items:
        row = item["row"]
        cropped = resize_output(item["image"][y : y + crop_h, x : x + crop_w])
        out_name = unique_output_name(location_dir, clean_stem(row.get("original_file_name", row["source_file"])))
        out_path = os.path.join(location_dir, out_name)
        cv2.imwrite(out_path, cropped, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
        rel_file = os.path.relpath(out_path, cfg.base)

        records.append(
            manifest_record_from_row(
                row,
                "aligned",
                file_name=rel_file,
                stats=item["stats"],
                crop_w=crop_w,
                crop_h=crop_h,
            )
        )

    print(f"  [SAVE] {len(records)} aligned images | crop=({x}, {y}, {crop_w}, {crop_h}) area={area_ratio:.2f} dropped={len(dropped)}")
    return records


def run_alignment(manifest_df: pd.DataFrame) -> pd.DataFrame:
    global alignment_failures, crop_drops, extractor, matcher, load_image
    alignment_failures = []
    crop_drops = []
    if cfg.redo_alignment:
        print("redo_alignment=True; deleting previous aligned outputs.")
        ensure_empty_dir(cfg.aligned_dir)
    elif alignment_complete():
        print("Alignment already complete; skipping.")
        return pd.read_csv(cfg.manifest_csv)
    else:
        os.makedirs(cfg.aligned_dir, exist_ok=True)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    LightGlue, SuperPoint, load_image = import_lightglue()
    extractor = SuperPoint(max_num_keypoints=cfg.max_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    df = manifest_df.copy()
    required_cols = {"source_file", "location", "time_of_day", "weather"}
    missing_cols = required_cols - set(df.columns)
    assert not missing_cols, f"Missing columns: {missing_cols}"

    if cfg.location_filter is not None:
        df = df[df["location"].astype(str).str.lower().str.strip().isin(cfg.location_filter)].copy()

    all_records = []
    for location, group in tqdm(list(df.groupby("location", sort=True)), desc="Align locations"):
        all_records.extend(align_location_group(location, group))

    all_records.extend(alignment_failures)
    all_records.extend(crop_drops)
    out_df = write_manifest(pd.DataFrame(all_records))
    print(f"\nAlignment stage wrote {len(out_df)} rows to {cfg.manifest_csv}")
    return out_df


def filtered_complete() -> bool:
    if not file_exists_and_nonempty(cfg.manifest_csv):
        return False
    try:
        df = pd.read_csv(cfg.manifest_csv)
    except Exception:
        return False
    if df.empty or "file_name" not in df.columns:
        return False
    filtered_rows = df[df["file_name"].astype(str).str.startswith("filtered_aligned/")]
    if filtered_rows.empty:
        return False
    missing = [
        p for p in filtered_rows["file_name"].astype(str) if not os.path.exists(os.path.join(cfg.base, p))
    ]
    if missing:
        print("Filtered files missing from existing manifest:", missing[:5])
        return False
    return True


def prepare_ecc_gray(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    scale = min(1.0, cfg.ecc_score_size / max(h, w))
    if scale < 1.0:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return gray.astype(np.float32) / 255.0


def ecc_warp_is_sane(warp: np.ndarray, shape) -> bool:
    h, w = shape[:2]
    corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, warp).reshape(-1, 2)
    area = abs(cv2.contourArea(warped.astype(np.float32)))
    src_area = max(1.0, float(w * h))
    scale = area / src_area
    if scale < cfg.ecc_min_scale or scale > cfg.ecc_max_scale:
        return False
    drift = np.linalg.norm(warped - corners.reshape(-1, 2), axis=1)
    max_allowed = cfg.ecc_max_corner_drift_frac * max(h, w)
    return float(drift.max()) <= max_allowed


def ecc_homography_score(reference_img: np.ndarray, moving_img: np.ndarray) -> float:
    ref_gray = prepare_ecc_gray(reference_img)
    mov_gray = prepare_ecc_gray(moving_img)
    if ref_gray.shape != mov_gray.shape:
        mov_gray = cv2.resize(mov_gray, (ref_gray.shape[1], ref_gray.shape[0]), interpolation=cv2.INTER_AREA)
    baseline_mse = float(np.mean((ref_gray - mov_gray) ** 2))

    warp = np.eye(3, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, cfg.ecc_max_iters, cfg.ecc_eps)
    try:
        cc, warp = cv2.findTransformECC(ref_gray, mov_gray, warp, cv2.MOTION_HOMOGRAPHY, criteria)
        if not ecc_warp_is_sane(warp, ref_gray.shape):
            return -baseline_mse
        warped = cv2.warpPerspective(
            mov_gray,
            warp,
            (ref_gray.shape[1], ref_gray.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        )
        mse = float(np.mean((ref_gray - warped) ** 2))
        if mse > baseline_mse * cfg.ecc_max_worse_factor:
            return -baseline_mse
        return float(cc) - mse
    except cv2.error:
        return -baseline_mse


def choose_best_duplicate(rows: list[pd.Series]):
    if len(rows) == 1:
        return rows[0], {"representative_score": 1.0}

    images = []
    usable_rows = []
    for row in rows:
        path = os.path.join(cfg.base, row["file_name"])
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            usable_rows.append(row)
            images.append(img)

    if len(usable_rows) == 1:
        return usable_rows[0], {"representative_score": 1.0}
    if not usable_rows:
        return rows[0], {"representative_score": float("nan")}

    scores = []
    for i, ref_img in enumerate(images):
        pair_scores = []
        for j, mov_img in enumerate(images):
            if i == j:
                continue
            pair_scores.append(ecc_homography_score(ref_img, mov_img))
        scores.append(float(np.mean(pair_scores)) if pair_scores else 1.0)

    best_idx = int(np.nanargmax(scores))
    return usable_rows[best_idx], {"representative_score": scores[best_idx]}


def run_filtered(aligned_df: pd.DataFrame) -> pd.DataFrame:
    if cfg.redo_filtered:
        print("redo_filtered=True; deleting previous filtered outputs.")
        ensure_empty_dir(cfg.filtered_dir)
    elif filtered_complete():
        print("Filtered dataset already complete; skipping.")
        return pd.read_csv(cfg.manifest_csv)
    else:
        os.makedirs(cfg.filtered_dir, exist_ok=True)

    if aligned_df.empty:
        return write_manifest(pd.DataFrame(columns=MANIFEST_COLUMNS))

    records = []
    aligned_candidates = aligned_df[aligned_df["status"] == "aligned"].copy() if "status" in aligned_df.columns else aligned_df.copy()
    non_aligned = aligned_df[aligned_df["status"] != "aligned"].copy() if "status" in aligned_df.columns else pd.DataFrame()
    records.extend([dict(row) for _, row in non_aligned.iterrows()])

    group_cols = ["location", "time_of_day", "weather"]
    for key, group in tqdm(list(aligned_candidates.groupby(group_cols, sort=True)), desc="Filter duplicates"):
        location, tod, weather = key
        rows = [row for _, row in group.iterrows()]
        selected, stats = choose_best_duplicate(rows)
        selected_source = selected["source_file"]

        src_path = os.path.join(cfg.base, selected["file_name"])
        img = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[MISS] filtered source missing: {src_path}")
            for row in rows:
                records.append(manifest_record_from_row(row, "filter_failed", "missing_aligned_file"))
            continue

        location_dir = os.path.join(cfg.filtered_dir, clean_stem(location))
        os.makedirs(location_dir, exist_ok=True)
        out_name = f"{clean_stem(location)}_{clean_stem(tod)}_{clean_stem(weather)}.jpg"
        out_path = os.path.join(location_dir, out_name)
        cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
        filtered_file = os.path.relpath(out_path, cfg.base)

        for row in rows:
            rec = dict(row)
            if row["source_file"] == selected_source:
                rec["file_name"] = filtered_file
                rec["caption"] = caption_from_labels(location, tod, weather)
                rec["representative_score"] = stats["representative_score"]
                rec["status"] = "kept"
                rec["drop_reason"] = ""
            else:
                rec["status"] = "duplicate_filtered"
                rec["drop_reason"] = "duplicate_condition"
                rec["representative_score"] = stats["representative_score"]
            records.append(rec)

    out_df = write_manifest(pd.DataFrame(records))
    print(f"\nFiltered stage wrote {len(out_df)} final rows to {cfg.manifest_csv}")
    return out_df


def hf_dataset_complete() -> bool:
    metadata_path = os.path.join(cfg.hf_dataset_dir, "metadata.jsonl")
    images_dir = os.path.join(cfg.hf_dataset_dir, "images")
    if not file_exists_and_nonempty(metadata_path) or not os.path.isdir(images_dir):
        return False
    try:
        meta = pd.read_json(metadata_path, lines=True)
    except Exception:
        return False
    if meta.empty or not {"file_name", "text"}.issubset(meta.columns):
        return False
    missing = [
        p for p in meta["file_name"].astype(str) if not os.path.exists(os.path.join(cfg.hf_dataset_dir, p))
    ]
    if missing:
        print("HF dataset files missing:", missing[:5])
        return False
    return True


def run_hf_export(final_manifest_df: pd.DataFrame) -> None:
    if cfg.redo_hf_dataset:
        print("redo_hf_dataset=True; deleting previous HF dataset export.")
        ensure_empty_dir(cfg.hf_dataset_dir)
    elif hf_dataset_complete():
        print("HF dataset already complete; skipping.")
        return
    else:
        os.makedirs(cfg.hf_dataset_dir, exist_ok=True)

    images_dir = os.path.join(cfg.hf_dataset_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    metadata_rows = []

    kept_rows = final_manifest_df[final_manifest_df["status"] == "kept"].copy() if "status" in final_manifest_df.columns else final_manifest_df.copy()
    for _, row in kept_rows.iterrows():
        src_path = os.path.join(cfg.base, row["file_name"])
        if not os.path.exists(src_path):
            print(f"[MISS] HF source missing: {src_path}")
            continue
        out_name = clean_stem(row["file_name"]) + ".jpg"
        dst_rel = os.path.join("images", out_name)
        dst_path = os.path.join(cfg.hf_dataset_dir, dst_rel)
        shutil.copy2(src_path, dst_path)
        metadata_rows.append(
            {
                "file_name": dst_rel,
                "text": row["caption"],
                "location": row["location"],
                "time_of_day": row["time_of_day"],
                "weather": row["weather"],
                "split": row["split"],
            }
        )

    metadata_path = os.path.join(cfg.hf_dataset_dir, "metadata.jsonl")
    with open(metadata_path, "w", encoding="utf-8") as f:
        for rec in metadata_rows:
            f.write(json.dumps(rec) + "\n")
    pd.DataFrame(metadata_rows).to_csv(os.path.join(cfg.hf_dataset_dir, "metadata.csv"), index=False)
    print(f"Exported {len(metadata_rows)} HF dataset rows to {cfg.hf_dataset_dir}")


def raw_image_count() -> int:
    if not os.path.isdir(cfg.raw_dir):
        return 0
    return sum(
        1
        for f in os.listdir(cfg.raw_dir)
        if os.path.splitext(f)[1].lower() in VALID_IMAGE_SUFFIXES and ":zone.identifier" not in f.lower()
    )


def print_final_summary(manifest_df: pd.DataFrame, aligned_df: pd.DataFrame, final_manifest_df: pd.DataFrame) -> None:
    filtered_count = int((final_manifest_df["status"] == "kept").sum()) if "status" in final_manifest_df.columns else len(final_manifest_df)
    hf_images = 0
    hf_images_dir = Path(cfg.hf_dataset_dir) / "images"
    if hf_images_dir.is_dir():
        hf_images = len(list(hf_images_dir.glob("*.jpg")))

    print("\nFinal dataset summary")
    print("---------------------")
    print("raw images:", raw_image_count())
    print("processed images:", len(manifest_df))
    print("aligned images:", len(aligned_df))
    print("filtered final images:", filtered_count)
    print("failed alignment count:", len(alignment_failures))
    print("dropped-for-crop-loss count:", len(crop_drops))
    print("hf dataset images:", hf_images)


def run_pipeline(config: PreprocessConfig) -> None:
    global cfg, device, torch
    cfg = config
    torch = import_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Base:", cfg.base)
    print("Raw images:", cfg.raw_dir)
    print("Processed images:", cfg.processed_dir)
    print("Aligned images:", cfg.aligned_dir)
    print("Filtered aligned images:", cfg.filtered_dir)
    print("Manifest CSV:", cfg.manifest_csv)
    print("HF dataset:", cfg.hf_dataset_dir)

    manifest_df = run_preprocess()
    summarize_manifest(manifest_df, "Preprocess summary", path_base=cfg.processed_dir)

    aligned_df = run_alignment(manifest_df)
    summarize_manifest(aligned_df, "Alignment summary", path_base=cfg.base)
    print("failed alignments:", len(alignment_failures))
    print("dropped for crop loss:", len(crop_drops))

    final_manifest_df = run_filtered(aligned_df)
    summarize_manifest(final_manifest_df, "Final manifest summary", path_base=cfg.base)

    run_hf_export(final_manifest_df)
    print_final_summary(manifest_df, aligned_df, final_manifest_df)


def parse_location_filter(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {part.strip().lower() for part in value.split(",") if part.strip()}


def default_base() -> str:
    candidates = [
        "/content/drive/MyDrive/CIS_5190_group_project",
        "/content/drive/My Drive/CIS_5190_group_project",
        os.path.abspath("data"),
    ]
    return next((p for p in candidates if os.path.exists(p)), candidates[-1])


def build_config(args: argparse.Namespace) -> PreprocessConfig:
    base = os.path.abspath(args.base)
    redo_preprocess = args.redo_all or args.redo_preprocess
    redo_alignment = args.redo_all or args.redo_alignment
    redo_filtered = args.redo_all or args.redo_filtered
    redo_hf_dataset = args.redo_all or args.redo_hf_dataset

    return PreprocessConfig(
        base=base,
        raw_dir=os.path.abspath(args.raw_dir) if args.raw_dir else os.path.join(base, "Images"),
        processed_dir=os.path.abspath(args.processed_dir) if args.processed_dir else os.path.join(base, "processedImages"),
        aligned_dir=os.path.abspath(args.aligned_dir) if args.aligned_dir else os.path.join(base, "aligned"),
        filtered_dir=os.path.abspath(args.filtered_dir) if args.filtered_dir else os.path.join(base, "filtered_aligned"),
        manifest_csv=os.path.abspath(args.manifest_csv) if args.manifest_csv else os.path.join(base, "manifest.csv"),
        hf_dataset_dir=os.path.abspath(args.hf_dataset_dir) if args.hf_dataset_dir else os.path.join(base, "hf_dataset"),
        redo_preprocess=redo_preprocess,
        redo_alignment=redo_alignment,
        redo_filtered=redo_filtered,
        redo_hf_dataset=redo_hf_dataset,
        location_filter=parse_location_filter(args.location_filter),
        output_size=args.output_size,
        jpeg_quality=args.jpeg_quality,
        resize=args.resize,
        max_keypoints=args.max_keypoints,
        ransac_reproj_threshold=args.ransac_reproj_threshold,
        min_matches=args.min_matches,
        min_inliers=args.min_inliers,
        min_inlier_ratio=args.min_inlier_ratio,
        min_shared_crop_area_ratio=args.min_shared_crop_area_ratio,
        ecc_score_size=args.ecc_score_size,
        ecc_max_iters=args.ecc_max_iters,
        ecc_eps=args.ecc_eps,
        ecc_max_worse_factor=args.ecc_max_worse_factor,
        ecc_max_corner_drift_frac=args.ecc_max_corner_drift_frac,
        ecc_min_scale=args.ecc_min_scale,
        ecc_max_scale=args.ecc_max_scale,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full preprocessing, alignment, filtering, and HF export pipeline.")
    parser.add_argument("--base", default=default_base(), help="Project data root containing Images/ and output folders.")
    parser.add_argument("--raw-dir", default=None, help="Raw image directory. Defaults to BASE/Images.")
    parser.add_argument("--processed-dir", default=None, help="Processed image directory. Defaults to BASE/processedImages.")
    parser.add_argument("--aligned-dir", default=None, help="Intermediate aligned image directory. Defaults to BASE/aligned.")
    parser.add_argument("--filtered-dir", default=None, help="Final filtered aligned image directory. Defaults to BASE/filtered_aligned.")
    parser.add_argument("--manifest-csv", default=None, help="Manifest CSV path. Defaults to BASE/manifest.csv.")
    parser.add_argument("--hf-dataset-dir", default=None, help="HF ImageFolder output directory. Defaults to BASE/hf_dataset.")
    parser.add_argument("--redo-all", action="store_true", help="Rebuild every pipeline stage.")
    parser.add_argument("--redo-preprocess", action="store_true", help="Rebuild processed images and initial manifest.")
    parser.add_argument("--redo-alignment", action="store_true", help="Rebuild aligned images and alignment manifest rows.")
    parser.add_argument("--redo-filtered", action="store_true", help="Rebuild filtered_aligned outputs.")
    parser.add_argument("--redo-hf-dataset", action="store_true", help="Rebuild HF dataset export.")
    parser.add_argument("--location-filter", default=None, help="Comma-separated location names for debugging.")
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--resize", type=int, default=1024, help="LightGlue image resize.")
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--ransac-reproj-threshold", type=float, default=5.0)
    parser.add_argument("--min-matches", type=int, default=12)
    parser.add_argument("--min-inliers", type=int, default=8)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.0)
    parser.add_argument("--min-shared-crop-area-ratio", type=float, default=0.50)
    parser.add_argument("--ecc-score-size", type=int, default=768)
    parser.add_argument("--ecc-max-iters", type=int, default=80)
    parser.add_argument("--ecc-eps", type=float, default=1e-5)
    parser.add_argument("--ecc-max-worse-factor", type=float, default=1.05)
    parser.add_argument("--ecc-max-corner-drift-frac", type=float, default=0.20)
    parser.add_argument("--ecc-min-scale", type=float, default=0.70)
    parser.add_argument("--ecc-max-scale", type=float, default=1.30)
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(build_config(parse_args()))
