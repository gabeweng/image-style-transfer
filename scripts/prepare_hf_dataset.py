"""
prepare_hf_dataset.py

Converts aligned_labels.csv into the HuggingFace ImageFolder format
expected by diffusers training scripts (LoRA and ControlNet).

Outputs
-------
Standard (for LoRA / SD img2img):
    data/hf_dataset/
    ├── metadata.jsonl
    └── images/
        └── *.jpg

ControlNet variant (--controlnet flag):
    data/hf_dataset_controlnet/
    ├── metadata.jsonl
    ├── images/           (source images)
    └── conditioning_images/  (Canny edge maps)

Usage
-----
    python scripts/prepare_hf_dataset.py \
        --aligned_csv  /path/to/aligned_labels.csv \
        --aligned_dir  /path/to/aligned \
        --output_dir   /path/to/data/hf_dataset

    # ControlNet variant:
    python scripts/prepare_hf_dataset.py \
        --aligned_csv  /path/to/aligned_labels.csv \
        --aligned_dir  /path/to/aligned \
        --output_dir   /path/to/data/hf_dataset_controlnet \
        --controlnet
"""

import argparse
import json
import os
import shutil

import cv2
import numpy as np
import pandas as pd
from PIL import Image


CAPTION_TEMPLATE = "A photo of {location} on the University of Pennsylvania campus at {tod}, {weather} weather"

# Canny thresholds — tune if edges are too sparse or noisy
CANNY_LOW = 100
CANNY_HIGH = 200


def make_caption(location: str, tod: str, weather: str) -> str:
    return CAPTION_TEMPLATE.format(
        location=location.replace("_", " ").title(),
        tod=tod.lower(),
        weather=weather.lower(),
    )


def extract_canny(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, CANNY_LOW, CANNY_HIGH)
    # 3-channel so diffusers conditioning pipeline accepts it
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def prepare(aligned_csv: str, aligned_dir: str, output_dir: str, controlnet: bool):
    df = pd.read_csv(aligned_csv)

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    if controlnet:
        cond_dir = os.path.join(output_dir, "conditioning_images")
        os.makedirs(cond_dir, exist_ok=True)

    records = []

    # Include both anchor (source condition) and warped targets as training examples
    seen_anchors = set()
    for _, row in df.iterrows():
        # Add anchor image once per location
        if row["anchor_file"] not in seen_anchors:
            seen_anchors.add(row["anchor_file"])
            src_path = os.path.join(aligned_dir, row["anchor_file"])
            if os.path.exists(src_path):
                dst_name = row["anchor_file"]
                shutil.copy2(src_path, os.path.join(images_dir, dst_name))
                caption = make_caption(row["location"], row["anchor_tod"], row["anchor_weather"])
                records.append({"file_name": f"images/{dst_name}", "text": caption})

                if controlnet:
                    edges = extract_canny(src_path)
                    cv2.imwrite(os.path.join(cond_dir, dst_name), edges)

        # Add warped target image
        src_path = os.path.join(aligned_dir, row["warped_path"])
        if not os.path.exists(src_path):
            print(f"[MISS] {src_path}")
            continue

        dst_name = row["warped_path"]
        shutil.copy2(src_path, os.path.join(images_dir, dst_name))
        caption = make_caption(row["location"], row["target_tod"], row["target_weather"])
        records.append({"file_name": f"images/{dst_name}", "text": caption})

        if controlnet:
            edges = extract_canny(src_path)
            cv2.imwrite(os.path.join(cond_dir, dst_name), edges)

    metadata_path = os.path.join(output_dir, "metadata.jsonl")
    with open(metadata_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Done. {len(records)} entries written to {output_dir}/")
    if controlnet:
        print(f"  Canny edge maps → {cond_dir}/")
    print(f"  metadata.jsonl  → {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned_csv", required=True)
    parser.add_argument("--aligned_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--controlnet", action="store_true",
                        help="Also generate Canny conditioning_images/ folder")
    args = parser.parse_args()

    prepare(args.aligned_csv, args.aligned_dir, args.output_dir, args.controlnet)
