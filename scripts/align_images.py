"""
align_images.py

Reads img_labels.csv, groups images by location, aligns each image to a
daytime-clear anchor using SIFT+FLANN homography, crops to the universal
overlap region, and writes aligned_labels.csv.

Usage (EC2 / local):
    python scripts/align_images.py \
        --images_dir  /path/to/processedImages \
        --labels_csv  /path/to/img_labels.csv \
        --output_dir  /path/to/aligned \
        --output_csv  /path/to/aligned_labels.csv \
        --size 512

Colab: set IMAGES_DIR / LABELS_CSV / OUTPUT_DIR / OUTPUT_CSV at the top of
       notebooks/02_align_and_audit.ipynb and call align_all() directly.
"""

import argparse
import os
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pillow_heif import register_heif_opener
from skimage.metrics import structural_similarity as ssim

register_heif_opener()

ANCHOR_TOD = "daytime"
ANCHOR_TOD_ALIASES = ("daytime", "day", "morning")
ANCHOR_WEATHER = "clear"


@dataclass
class AlignConfig:
    lowe_ratio: float = 0.7
    ransac_thresh: float = 4.0
    min_good_matches: int = 40
    min_inliers: int = 25
    min_inlier_ratio: float = 0.25
    max_scale_change: float = 4.0
    max_rotation_deg: float = 45.0
    sift_features: int = 8000
    flann_checks: int = 100
    anchor_strategy: str = "condition"
    match_mode: str = "gray"
    canny_low: int = 60
    canny_high: int = 180
    max_drift: float | None = None
    drift_grid_step: int = 16


def load_image_bgr(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def prepare_match_image(gray: np.ndarray, config: AlignConfig) -> np.ndarray:
    mode = config.match_mode
    if mode == "gray":
        return gray

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(gray)
    if mode == "clahe":
        return normalized

    blurred = cv2.GaussianBlur(normalized, (3, 3), 0)
    edges = cv2.Canny(blurred, config.canny_low, config.canny_high)
    if mode == "edges":
        return edges
    if mode == "clahe_edges":
        return cv2.addWeighted(normalized, 0.65, edges, 0.35, 0)

    raise ValueError(f"Unsupported match_mode: {mode}")


def pick_anchor(group: pd.DataFrame) -> pd.Series | None:
    tod_col = group["time_of_day"].str.lower()
    wx_col = group["weather"].str.lower()
    mask = tod_col.isin(ANCHOR_TOD_ALIASES) & (wx_col == ANCHOR_WEATHER)
    candidates = group[mask]
    if candidates.empty:
        # Fall back: any bright/daylike shot
        candidates = group[tod_col.isin(ANCHOR_TOD_ALIASES)]
    if candidates.empty:
        return None
    return candidates.iloc[0]


def compute_homography(ref_gray, img_gray, config: AlignConfig):
    ref_match = prepare_match_image(ref_gray, config)
    img_match = prepare_match_image(img_gray, config)

    sift = cv2.SIFT_create(nfeatures=config.sift_features)
    kp_ref, des_ref = sift.detectAndCompute(ref_match, None)
    kp_img, des_img = sift.detectAndCompute(img_match, None)

    if des_ref is None or des_img is None:
        return None, {
            "good_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "reason": "missing_descriptors",
        }

    # RootSIFT improves match distinctiveness under lighting changes.
    des_ref = rootsift(des_ref)
    des_img = rootsift(des_img)

    flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": config.flann_checks})
    matches_fwd = flann.knnMatch(des_ref, des_img, k=2)
    matches_rev = flann.knnMatch(des_img, des_ref, k=2)

    good_fwd = ratio_filter(matches_fwd, config.lowe_ratio)
    good_rev = ratio_filter(matches_rev, config.lowe_ratio)

    # Mutual nearest-neighbor check removes many one-off false correspondences.
    rev_pairs = {(m.trainIdx, m.queryIdx) for m in good_rev}
    good = [m for m in good_fwd if (m.queryIdx, m.trainIdx) in rev_pairs]
    before_drift_filter = len(good)

    if config.max_drift is not None and config.max_drift > 0:
        good = [
            m for m in good
            if np.linalg.norm(
                np.array(kp_ref[m.queryIdx].pt) - np.array(kp_img[m.trainIdx].pt)
            ) <= config.max_drift
        ]

    if len(good) < config.min_good_matches:
        return None, {
            "good_matches": len(good),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "reason": "not_enough_good_matches",
            "drift_filtered_matches": before_drift_filter - len(good),
        }

    src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_img[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(
        dst_pts,
        src_pts,
        cv2.RANSAC,
        config.ransac_thresh,
        maxIters=5000,
        confidence=0.995,
    )
    if M is None or mask is None:
        return None, {
            "good_matches": len(good),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "reason": "homography_failed",
            "drift_filtered_matches": before_drift_filter - len(good),
        }

    inliers = int(mask.ravel().sum())
    inlier_ratio = inliers / len(good)
    if inliers < config.min_inliers or inlier_ratio < config.min_inlier_ratio:
        return None, {
            "good_matches": len(good),
            "inliers": inliers,
            "inlier_ratio": inlier_ratio,
            "reason": "too_few_inliers",
            "drift_filtered_matches": before_drift_filter - len(good),
        }

    sane, reason = homography_is_sane(M, img_gray.shape, ref_gray.shape, config)
    if not sane:
        return None, {
            "good_matches": len(good),
            "inliers": inliers,
            "inlier_ratio": inlier_ratio,
            "reason": reason,
            "drift_filtered_matches": before_drift_filter - len(good),
        }

    return M, {
        "good_matches": len(good),
        "inliers": inliers,
        "inlier_ratio": inlier_ratio,
        "reason": "ok",
        "drift_filtered_matches": before_drift_filter - len(good),
    }


def rootsift(des: np.ndarray) -> np.ndarray:
    des = des.astype(np.float32)
    des /= des.sum(axis=1, keepdims=True) + 1e-7
    return np.sqrt(des)


def ratio_filter(matches, lowe_ratio: float):
    good = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < lowe_ratio * n.distance:
            good.append(m)
    return good


def homography_is_sane(
    M: np.ndarray,
    src_shape: tuple[int, int],
    dst_shape: tuple[int, int],
    config: AlignConfig,
) -> tuple[bool, str]:
    src_h, src_w = src_shape
    dst_h, dst_w = dst_shape
    corners = np.float32([
        [0, 0],
        [src_w - 1, 0],
        [src_w - 1, src_h - 1],
        [0, src_h - 1],
    ]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, M).reshape(-1, 2)
    area = abs(cv2.contourArea(warped.astype(np.float32)))
    src_area = src_w * src_h
    if area <= 0:
        return False, "degenerate_warp"
    scale = area / src_area
    if scale < 1 / config.max_scale_change or scale > config.max_scale_change:
        return False, "scale_out_of_bounds"

    # Estimate local rotation from the top edge. This catches dramatic wrong flips.
    dx, dy = warped[1] - warped[0]
    angle = abs(np.degrees(np.arctan2(dy, dx)))
    angle = min(angle, abs(180 - angle))
    if angle > config.max_rotation_deg:
        return False, "rotation_out_of_bounds"

    if config.max_drift is not None and config.max_drift > 0:
        step = max(1, int(config.drift_grid_step))
        xs = np.unique(np.concatenate([np.arange(0, src_w, step), [src_w - 1]])).astype(np.float32)
        ys = np.unique(np.concatenate([np.arange(0, src_h, step), [src_h - 1]])).astype(np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        reference_points = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)
        warped_points = cv2.perspectiveTransform(reference_points.reshape(-1, 1, 2), M).reshape(-1, 2)
        drift = np.linalg.norm(warped_points - reference_points, axis=1)
        if float(drift.max()) > config.max_drift:
            return False, "drift_out_of_bounds"

    # Require at least some overlap with the reference canvas.
    x0, y0 = warped.min(axis=0)
    x1, y1 = warped.max(axis=0)
    overlap_w = max(0, min(x1, dst_w) - max(x0, 0))
    overlap_h = max(0, min(y1, dst_h) - max(y0, 0))
    if overlap_w * overlap_h < 0.1 * dst_w * dst_h:
        return False, "low_canvas_overlap"
    return True, "ok"


def align_location_group(
    group: pd.DataFrame,
    images_dir: str,
    output_dir: str,
    target_size: tuple[int, int],
    config: AlignConfig,
) -> list[dict]:
    anchor_row = pick_best_anchor(group, images_dir, config) if config.anchor_strategy == "best" else pick_anchor(group)
    if anchor_row is None:
        print(f"  [SKIP] No anchor candidate for location '{group.iloc[0]['location']}'")
        return []

    anchor_path = os.path.join(images_dir, anchor_row["file_name"])
    if not os.path.exists(anchor_path):
        print(f"  [SKIP] Anchor file not found: {anchor_path}")
        return []

    anchor_bgr = load_image_bgr(anchor_path)
    h, w = anchor_bgr.shape[:2]
    anchor_gray = cv2.cvtColor(anchor_bgr, cv2.COLOR_BGR2GRAY)

    master_mask = np.ones((h, w), dtype=np.uint8) * 255
    records = []

    for _, row in group.iterrows():
        if row["file_name"] == anchor_row["file_name"]:
            continue

        target_path = os.path.join(images_dir, row["file_name"])
        if not os.path.exists(target_path):
            print(f"  [MISS] {row['file_name']}")
            continue

        target_bgr = load_image_bgr(target_path)
        target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)

        M, stats = compute_homography(anchor_gray, target_gray, config)
        if M is None:
            print(
                f"  [FAIL] homography for {row['file_name']} "
                f"(good={stats['good_matches']}, inliers={stats['inliers']}, "
                f"ratio={stats['inlier_ratio']:.2f}, "
                f"drift_filtered={stats.get('drift_filtered_matches', 0)}, "
                f"reason={stats['reason']})"
            )
            continue

        warped = cv2.warpPerspective(target_bgr, M, (w, h))

        warped_mask = cv2.threshold(
            cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY
        )[1]
        master_mask = cv2.bitwise_and(master_mask, warped_mask)

        records.append({
            "row": row,
            "warped": warped,
            "stats": stats,
        })

    if not records:
        return []

    coords = cv2.findNonZero(master_mask)
    if coords is None:
        print(f"  [SKIP] No common overlap for '{anchor_row['location']}'")
        return []

    x, y, cw, ch = cv2.boundingRect(coords)
    anchor_cropped = cv2.resize(anchor_bgr[y:y+ch, x:x+cw], target_size)

    out_anchor = f"{anchor_row['location']}_anchor.jpg"
    cv2.imwrite(os.path.join(output_dir, out_anchor), anchor_cropped)

    results = []
    for rec in records:
        row = rec["row"]
        warped = rec["warped"]
        cropped = cv2.resize(warped[y:y+ch, x:x+cw], target_size)

        stem = os.path.splitext(os.path.basename(row["file_name"]))[0]
        out_name = f"{stem}_aligned.jpg"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, cropped)

        anchor_gray_crop = cv2.cvtColor(anchor_cropped, cv2.COLOR_BGR2GRAY)
        warped_gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        ssim_score = float(ssim(anchor_gray_crop, warped_gray_crop, data_range=255))

        results.append({
            "location": row["location"],
            "anchor_file": out_anchor,
            "anchor_tod": anchor_row["time_of_day"],
            "anchor_weather": anchor_row["weather"],
            "target_file": row["file_name"],
            "target_tod": row["time_of_day"],
            "target_weather": row["weather"],
            "warped_path": out_name,
            "ssim_score": round(ssim_score, 4),
            "n_matches": rec["stats"]["good_matches"],
            "n_inliers": rec["stats"]["inliers"],
            "inlier_ratio": round(rec["stats"]["inlier_ratio"], 4),
            "drift_filtered_matches": rec["stats"].get("drift_filtered_matches", 0),
        })
        print(
            f"  [OK] {row['file_name']} → SSIM={ssim_score:.3f}, "
            f"good={rec['stats']['good_matches']}, "
            f"inliers={rec['stats']['inliers']}, "
            f"ratio={rec['stats']['inlier_ratio']:.2f}, "
            f"drift_filtered={rec['stats'].get('drift_filtered_matches', 0)}"
        )

    return results


def pick_best_anchor(group: pd.DataFrame, images_dir: str, config: AlignConfig) -> pd.Series | None:
    cache = {}
    rows = list(group.iterrows())
    best_idx = None
    best_score = -1

    for idx_ref, ref_row in rows:
        ref_path = os.path.join(images_dir, ref_row["file_name"])
        if not os.path.exists(ref_path):
            continue
        if idx_ref not in cache:
            ref_bgr = load_image_bgr(ref_path)
            cache[idx_ref] = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
        ref_gray = cache[idx_ref]

        score = 0
        for idx_img, img_row in rows:
            if idx_img == idx_ref:
                continue
            img_path = os.path.join(images_dir, img_row["file_name"])
            if not os.path.exists(img_path):
                continue
            if idx_img not in cache:
                img_bgr = load_image_bgr(img_path)
                cache[idx_img] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            _, stats = compute_homography(ref_gray, cache[idx_img], config)
            score += stats["inliers"]

        if score > best_score:
            best_idx = idx_ref
            best_score = score

    if best_idx is None:
        return None

    row = group.loc[best_idx]
    print(f"  [ANCHOR] selected {row['file_name']} by best connectivity (score={best_score})")
    return row


def align_all(
    images_dir: str,
    labels_csv: str,
    output_dir: str,
    output_csv: str,
    size: int = 512,
    config: AlignConfig | None = None,
):
    os.makedirs(output_dir, exist_ok=True)
    config = config or AlignConfig()
    df = pd.read_csv(labels_csv)
    df["time_of_day"] = df["time_of_day"].str.lower().str.strip()
    df["weather"] = df["weather"].str.lower().str.strip()

    all_records = []
    for location, group in df.groupby("location"):
        print(f"\nProcessing location: {location} ({len(group)} images)")
        recs = align_location_group(group, images_dir, output_dir, (size, size), config)
        all_records.extend(recs)

    out_df = pd.DataFrame(all_records)
    out_df.to_csv(output_csv, index=False)
    print(f"\nDone. {len(out_df)} aligned pairs written to {output_csv}")
    return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True, help="Folder with processed images")
    parser.add_argument("--labels_csv", required=True, help="Path to img_labels.csv")
    parser.add_argument("--output_dir", required=True, help="Folder to write aligned images")
    parser.add_argument("--output_csv", required=True, help="Path to write aligned_labels.csv")
    parser.add_argument("--size", type=int, default=512, help="Output image size (square)")
    parser.add_argument("--lowe_ratio", type=float, default=0.7)
    parser.add_argument("--ransac_thresh", type=float, default=4.0)
    parser.add_argument("--min_good_matches", type=int, default=40)
    parser.add_argument("--min_inliers", type=int, default=25)
    parser.add_argument("--min_inlier_ratio", type=float, default=0.25)
    parser.add_argument("--max_scale_change", type=float, default=4.0)
    parser.add_argument("--max_rotation_deg", type=float, default=45.0)
    parser.add_argument("--sift_features", type=int, default=8000)
    parser.add_argument("--flann_checks", type=int, default=100)
    parser.add_argument(
        "--match_mode",
        choices=["gray", "clahe", "edges", "clahe_edges"],
        default="gray",
        help="Image representation used for SIFT matching. Try clahe_edges for day/night pairs.",
    )
    parser.add_argument("--canny_low", type=int, default=60)
    parser.add_argument("--canny_high", type=int, default=180)
    parser.add_argument(
        "--max_drift",
        type=float,
        default=None,
        help="Maximum allowed pixel movement across a dense image grid. Set 0 or omit to disable.",
    )
    parser.add_argument(
        "--drift_grid_step",
        type=int,
        default=16,
        help="Pixel spacing for max_drift checks. Use 1 to check every pixel exactly; larger values are faster.",
    )
    parser.add_argument(
        "--anchor_strategy",
        choices=["condition", "best"],
        default="condition",
        help="'condition' uses daytime/clear fallback logic; 'best' selects the image with strongest pairwise connectivity.",
    )
    args = parser.parse_args()

    config = AlignConfig(
        lowe_ratio=args.lowe_ratio,
        ransac_thresh=args.ransac_thresh,
        min_good_matches=args.min_good_matches,
        min_inliers=args.min_inliers,
        min_inlier_ratio=args.min_inlier_ratio,
        max_scale_change=args.max_scale_change,
        max_rotation_deg=args.max_rotation_deg,
        sift_features=args.sift_features,
        flann_checks=args.flann_checks,
        anchor_strategy=args.anchor_strategy,
        match_mode=args.match_mode,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        max_drift=args.max_drift,
        drift_grid_step=args.drift_grid_step,
    )
    align_all(args.images_dir, args.labels_csv, args.output_dir, args.output_csv, args.size, config)
