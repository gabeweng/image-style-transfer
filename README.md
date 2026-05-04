# Image Style Transfer — CIS 4190/5190

Transform UPenn campus images across time-of-day and weather conditions while preserving scene geometry.

## Repository Layout

```
image-style-transfer/
├── notebooks/
│   ├── 00_preprocess_and_align.ipynb  # main data-prep notebook
│   ├── 01_pipeline_colab.ipynb        # optional post-dataset training runner
│   ├── 02_inference.ipynb             # inference experiments
│   └── 03_evaluate.ipynb              # evaluation experiments
├── scripts/
│   ├── preprocess.py                # full cluster/local data-prep CLI
│   ├── audit_app.py
│   └── train_classifier.py
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

## Setup

Local setup:

```bash
uv sync
```

Without `uv`:

```bash
pip install -r requirements.txt
```

The Colab notebooks install their own runtime dependencies.

## Data Layout

The active preprocessing pipeline expects the project data to live in Google Drive:

```
CIS_5190_group_project/
├── Images/              raw uploaded photos
├── processedImages/     orientation-corrected JPEGs
├── aligned/             intermediate aligned images
├── filtered_aligned/    final one-per-condition aligned images
├── hf_dataset/          Hugging Face ImageFolder export
├── checkpoints/
├── outputs/
└── manifest.csv
```

Raw filenames should follow:

```
LOCATION_TIME_WEATHER[_N].EXT
```

Examples:

```
agh3rd_day_cloudy.jpg
arch_night_clear_2.HEIC
vp_sunset_cloudy.JPG
castle_night_clear_ai.png
```

The optional trailing `_ai` marks synthetic images and is recorded in `manifest.csv` as `is_synthetic=True`. The preprocessing pipeline parses `location`, `time_of_day`, and `weather` from the filename, generates stable captions, and records every uploaded image in `manifest.csv`.

## Main Preprocessing Workflow

Run [notebooks/00_preprocess_and_align.ipynb](notebooks/00_preprocess_and_align.ipynb) in Colab for visual debugging, or run the equivalent cluster/local CLI:

```bash
python scripts/preprocess.py \
  --base /path/to/CIS_5190_group_project \
  --redo-all
```

Both paths produce the same core outputs: `processedImages/`, `aligned/`, `filtered_aligned/`, `manifest.csv`, and `hf_dataset/`.

The preprocessing pipeline performs the full sequence:

1. Converts raw uploads into standardized JPEGs under `processedImages/<location>/<time>_<weather>/`.
2. Builds and updates `manifest.csv`.
3. Aligns each location group with feature matching and homography.
4. Crops black or invalid warp areas using a uniform group crop.
5. Drops images when the shared crop would lose too much image area.
6. Filters duplicate images so each `location + time_of_day + weather` condition keeps one representative.
7. Writes final images under `filtered_aligned/`.
8. Exports `hf_dataset/images/` and `hf_dataset/metadata.jsonl` for diffusion fine-tuning.

The main notebook control flags are in the configuration cell:

```python
REDO_PREPROCESS = True
REDO_ALIGNMENT = True
REDO_FILTERED = True
REDO_HF_DATASET = True
OUTPUT_SIZE = 512
MIN_SHARED_CROP_AREA_RATIO = 0.50
```

When a `REDO_*` flag is `True`, the corresponding output folder is replaced. When it is `False`, the notebook checks whether the expected files already exist and skips completed work; rerun stages create missing directories and use unique output names instead of deleting existing image folders.

The script exposes equivalent flags:

```bash
python scripts/preprocess.py --base /path/to/project --redo-preprocess
python scripts/preprocess.py --base /path/to/project --redo-alignment
python scripts/preprocess.py --base /path/to/project --redo-filtered
python scripts/preprocess.py --base /path/to/project --redo-hf-dataset
```

## Manifest

`manifest.csv` is the source of truth for preprocessing outcomes. It contains all uploaded images, including failed or dropped rows.

Important columns:

```
file_name
location
time_of_day
weather
is_synthetic
caption
source_file
original_file_name
anchor_file
crop_w
crop_h
crop_area_ratio
matches
inliers
inlier_ratio
representative_score
split
status
drop_reason
```

Use `status == "kept"` for the final training images. Other statuses explain what happened to non-final images, such as `alignment_failed`, `crop_dropped`, or `duplicate_filtered`.

For alignment and crop auditing, `anchor_file` records the processed image used as the location-group anchor, and `crop_area_ratio` records the retained shared crop area relative to that anchor. These fields make it easier to review `crop_dropped` and `alignment_failed` rows later.

## Hugging Face Dataset

The final diffusion dataset is exported by `00_preprocess_and_align.ipynb` or `scripts/preprocess.py`:

```
hf_dataset/
├── metadata.jsonl
└── images/
    └── *.jpg
```

`metadata.jsonl` uses the standard ImageFolder format:

```json
{"file_name": "images/agh3rd_day_cloudy.jpg", "text": "A photo of Agh3rd on the University of Pennsylvania campus at day, cloudy weather"}
```

Only rows with `status == "kept"` are exported.

## Optional Training

After preprocessing finishes, [notebooks/01_pipeline_colab.ipynb](notebooks/01_pipeline_colab.ipynb) can validate the manifest/HF dataset and run optional LoRA or ControlNet training from Colab.

For LoRA training, the relevant dataset path is:

```
CIS_5190_group_project/hf_dataset
```

The notebook writes training checkpoints under:

```
CIS_5190_group_project/checkpoints/
```

## Notes

The old separate preprocessing/alignment notebooks and stale CSV-based scripts have been removed from the active workflow. Use `00_preprocess_and_align.ipynb` for Colab inspection or `scripts/preprocess.py` for cluster/local runs so teammates all use the same manifest-based pipeline.


## Manual Auditing

After pre-processing, you can (optionally) manually audit the hf_dataset output by running this line
```sh
streamlit run scripts/audit_app.py -- \
        --dataset_dir hf_dataset \
        --manifest_csv hf_dataset/metadata.csv \
        --output_csv hf_dataset/audit_decisions.csv
```