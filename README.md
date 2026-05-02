# Image Style Transfer — CIS 4190/5190

Transform UPenn campus images across time-of-day (daytime / sunset / night) and weather (sunny / cloudy / rainy) conditions while preserving scene geometry.

## Repository layout

```
image-style-transfer/
├── notebooks/
│   ├── 01_preprocessing.ipynb      # HEIC→JPEG, build img_labels.csv
│   ├── 02_inference.ipynb          # all four models: SD baseline, IP2P, ControlNet, ControlNet+LoRA
│   └── 03_evaluate.ipynb           # LPIPS + Condition Accuracy comparison table
├── scripts/
│   ├── align_images.py             # standalone alignment (run on EC2)
│   ├── audit_app.py                # Streamlit pair-curation UI
│   ├── prepare_hf_dataset.py       # convert aligned CSV → HF ImageFolder format
│   └── train_classifier.py         # ResNet-18 condition classifier (run on EC2)
├── Dockerfile
└── requirements.txt
```

---

## Setup

### Local / Colab

```bash
pip install -r requirements.txt
```

### EC2 (Docker)

```bash
# Build image
docker build -t image-style-transfer .

# Run with volume-mapped workspace
docker run --gpus all -it \
  -v /home/ubuntu/project:/workspace \
  image-style-transfer
```

---

## Data

All raw images (`*.HEIC`, `*.JPG`) and CSV files live on Google Drive:
```
CIS_5190_group_project/
├── Images/              raw photos
├── processedImages/     center-cropped outputs from notebook 01
├── aligned/             homography-aligned outputs from align_images.py
├── img_labels.csv
├── aligned_labels.csv
└── lpips_eval_set.csv
```

### Filename convention
```
LOCATION_TOD_WEATHER_N.EXT
```
- `LOCATION` — e.g. `LOCUSTWALK`, `GREGORY`, `HARRISON`
- `TOD` — `daytime` | `sunset` | `night`
- `WEATHER` — `sunny` | `cloudy` | `rainy` | `clear`
- `N` — integer index

### HuggingFace dataset format (required for fine-tuning)

Run this once after alignment to produce the training dataset:

```bash
# Standard (LoRA / SD img2img)
python scripts/prepare_hf_dataset.py \
    --aligned_csv  /path/to/aligned_labels.csv \
    --aligned_dir  /path/to/aligned \
    --output_dir   /path/to/data/hf_dataset

# ControlNet variant (also generates Canny conditioning_images/)
python scripts/prepare_hf_dataset.py \
    --aligned_csv  /path/to/aligned_labels.csv \
    --aligned_dir  /path/to/aligned \
    --output_dir   /path/to/data/hf_dataset_controlnet \
    --controlnet
```

Output layout:
```
hf_dataset/
├── metadata.jsonl          {"file_name": "images/foo.jpg", "text": "A photo of..."}
└── images/
    └── *.jpg

hf_dataset_controlnet/
├── metadata.jsonl
├── images/
└── conditioning_images/    Canny edge maps (same filenames as images/)
```

---

## Data pipeline

### 1. Preprocess
Run `notebooks/01_preprocessing.ipynb` on Colab (mounts Drive, reads `Images/`, writes `processedImages/` and `img_labels.csv`).

### 2. Align
```bash
# EC2 or Colab
python scripts/align_images.py \
    --images_dir /path/to/processedImages \
    --labels_csv /path/to/img_labels.csv \
    --output_dir /path/to/aligned \
    --output_csv /path/to/aligned_labels.csv \
    --size 512
```

### 3. Audit
```bash
# Local only — opens a browser at http://localhost:8501
streamlit run scripts/audit_app.py -- \
    --aligned_csv /path/to/aligned_labels.csv \
    --aligned_dir /path/to/aligned \
    --eval_csv    /path/to/lpips_eval_set.csv
```
Approve pairs to populate `lpips_eval_set.csv` (the strict LPIPS evaluation set).

---

## Fine-tuning (EC2 — run inside tmux)

### Train condition classifier
```bash
python scripts/train_classifier.py \
    --images_dir /workspace/data/aligned \
    --labels_csv /workspace/data/aligned_labels.csv \
    --output_dir /workspace/checkpoints \
    --epochs 20 \
    --batch_size 32
```
Saves `checkpoints/classifier_best.pt`.

### LoRA fine-tuning on Penn campus images
```bash
accelerate launch \
  diffusers_examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --train_data_dir="/workspace/data/hf_dataset" \
  --output_dir="/workspace/checkpoints/lora" \
  --resolution=512 \
  --train_batch_size=4 \
  --num_train_epochs=10 \
  --learning_rate=1e-4 \
  --lr_scheduler="cosine" \
  --mixed_precision="fp16" \
  --gradient_checkpointing \
  --checkpointing_steps=500 \
  --caption_column="text"
```

### ControlNet fine-tuning
```bash
accelerate launch \
  diffusers_examples/controlnet/train_controlnet.py \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --output_dir="/workspace/checkpoints/controlnet" \
  --train_data_dir="/workspace/data/hf_dataset_controlnet" \
  --resolution=512 \
  --train_batch_size=2 \
  --num_train_epochs=5 \
  --mixed_precision="fp16" \
  --gradient_checkpointing \
  --conditioning_image_column="conditioning_images" \
  --image_column="images" \
  --caption_column="text"
```

---

## Inference + Evaluation

Run `notebooks/02_inference.ipynb` — toggle `RUN_*` flags at the top to choose which models to run. All four models run sequentially with GPU memory cleared between each. Set `LORA_DIR = None` or `RUN_CONTROLNET_LORA = False` to skip model D before LoRA training is done.

Then run `notebooks/03_evaluate.ipynb` to compute LPIPS and Condition Accuracy across all models.

| Metric | Description |
|--------|-------------|
| **LPIPS ↓** | Perceptual similarity to ground-truth (AlexNet, `lpips` library) |
| **Condition Accuracy ↑** | % of generated images classified as target condition by `classifier_best.pt` |

---

## Reproducibility

```bash
# Full pipeline (EC2)
python scripts/align_images.py        [args]
python scripts/prepare_hf_dataset.py  [args]
python scripts/train_classifier.py    [args]
# kick off LoRA / ControlNet training (see Fine-tuning section)
# run notebooks/02_inference.ipynb
# run notebooks/03_evaluate.ipynb
```
