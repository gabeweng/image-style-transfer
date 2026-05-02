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

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pillow_heif import register_heif_opener
from skimage.metrics import structural_similarity as ssim

register_heif_opener()

ANCHOR_TOD = "daytime"
ANCHOR_WEATHER = "clear"
MIN_GOOD_MATCHES = 10
RANSAC_THRESH = 5.0
LOWE_RATIO = 0.7


def load_image_bgr(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def pick_anchor(group: pd.DataFrame) -> pd.Series | None:
    tod_col = group["time_of_day"].str.lower()
    wx_col = group["weather"].str.lower()
    mask = (tod_col == ANCHOR_TOD) & (wx_col == ANCHOR_WEATHER)
    candidates = group[mask]
    if candidates.empty:
        # Fall back: any daytime shot
        candidates = group[tod_col == ANCHOR_TOD]
    if candidates.empty:
        return None
    return candidates.iloc[0]


def compute_homography(ref_gray, img_gray):
    sift = cv2.SIFT_create()
    kp_ref, des_ref = sift.detectAndCompute(ref_gray, None)
    kp_img, des_img = sift.detectAndCompute(img_gray, None)

    if des_ref is None or des_img is None:
        return None, 0

    flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50})
    matches = flann.knnMatch(des_ref, des_img, k=2)
    good = [m for m, n in matches if m.distance < LOWE_RATIO * n.distance]

    if len(good) < MIN_GOOD_MATCHES:
        return None, len(good)

    src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_img[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, RANSAC_THRESH)
    return M, len(good)


def align_location_group(
    group: pd.DataFrame,
    images_dir: str,
    output_dir: str,
    target_size: tuple[int, int],
) -> list[dict]:
    anchor_row = pick_anchor(group)
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

        M, n_matches = compute_homography(anchor_gray, target_gray)
        if M is None:
            print(f"  [FAIL] homography for {row['file_name']} ({n_matches} matches)")
            continue

        warped = cv2.warpPerspective(target_bgr, M, (w, h))

        warped_mask = cv2.threshold(
            cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY
        )[1]
        master_mask = cv2.bitwise_and(master_mask, warped_mask)

        records.append({
            "row": row,
            "warped": warped,
            "n_matches": n_matches,
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

        stem = os.path.splitext(row["file_name"])[0]
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
            "n_matches": rec["n_matches"],
        })
        print(f"  [OK] {row['file_name']} → SSIM={ssim_score:.3f}, matches={rec['n_matches']}")

    return results


def align_all(images_dir: str, labels_csv: str, output_dir: str, output_csv: str, size: int = 512):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(labels_csv)
    df["time_of_day"] = df["time_of_day"].str.lower().str.strip()
    df["weather"] = df["weather"].str.lower().str.strip()

    all_records = []
    for location, group in df.groupby("location"):
        print(f"\nProcessing location: {location} ({len(group)} images)")
        recs = align_location_group(group, images_dir, output_dir, (size, size))
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
    args = parser.parse_args()

    align_all(args.images_dir, args.labels_csv, args.output_dir, args.output_csv, args.size)
