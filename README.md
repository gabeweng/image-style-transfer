# Image Style Transfer — CIS 4190/5190

Transform UPenn campus images across time-of-day and weather conditions while preserving scene geometry.

## Repository Layout

```
image-style-transfer/
├── notebooks/
│   ├── 00_preprocess_and_align.ipynb  # main data-prep notebook
│   ├── 01_training.ipynb               # train LoRA weights
│   ├── 02_inference.ipynb             # inference experiments
│   ├── 03_evaluate.ipynb              # evaluation experiments
│   └── 04_single_image.ipynb           # runs models on single image
├── scripts/
│   ├── preprocess.py                # script version of 00_preprocess...
│   ├── audit_app.py                 # human audit preprocessed steps
│   └── train_classifier.py          # train img->condition classifier
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

## Setup

This assumes we have access to `image-style-transfer`.

### Local setup (MacOS/Linux):

```bash
cd image-style-transfer
uv sync
# Without uv, run the line below
pip install -r requirements.txt
```

### Colab Setup
The Colab notebooks install their own runtime dependencies. Note that the active preprocessing pipeline expects that within `MyDrive` folder of Google Drive, we have a folder called `CIS_5190_group_project/Images` that has all the images within our training set.

The Colab notebooks install their own runtime dependencies with `uv pip install --system` before mounting Google Drive, so all you'd need to do is run the Colab. If you're on Colab, run the notebooks in numerical order (00, 01, 02, 03) and it should work. Ignore all other running instructions below. 


**Note**: The full pipeline hasn't been tested on Colab.

### Windows (Docker Desktop + GPU)

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/) with the WSL2 backend enabled and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed inside WSL2.

One-time WSL2 setup:
```bash
# Run inside WSL2 (wsl in PowerShell)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
```
Then restart Docker Desktop. Verify with: `docker run --gpus all --rm nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`

```powershell
# Build (run once, or after changing requirements.txt)
docker build -t image-style-transfer .

# Interactive shell with data mounted
docker run --gpus all -it `
  -v "C:\path\to\your\data:/workspace/data" `
  image-style-transfer
```

### EC2 (Docker — full training)

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
├── Images/              raw uploaded photos
├── processedImages/     orientation-corrected JPEGs
├── aligned/             intermediate aligned images
├── filtered_aligned/    final one-per-condition aligned images
├── hf_dataset/          Hugging Face ImageFolder export
├── checkpoints/         Model checkpoints
├── outputs/             Model validation outputs
└── manifest.csv         Info on model data
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

```bash
python scripts/preprocess.py \
  --base /path/to/image-style-transfer \
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


### Manifest

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

## Manual Auditing

After pre-processing, you can (optionally) manually audit the hf_dataset output by running this line
```bash
streamlit run scripts/audit_app.py -- \
        --dataset_dir hf_dataset \
        --manifest_csv hf_dataset/metadata.csv \
        --output_csv audit_decisions.csv
```


## Run the Training/Inference/Evaluate Steps
We used Vscode to execute the inference/evaluate notebooks on the EC2 instances/locally, but to run it without Vscode, you can install `papermill` to run the notebooks on the command line and get the output files.  

```bash
uv add papermill
cd notebooks/
papermill 01_training.ipynb out1.ipynb --log_output
papermill 02_inference.ipynb out2.ipynb --log-output
papermill 03_evaluate.ipynb out3.ipynb --log-output
```


## TA Validation 
For `05_ta_validation.ipynb`, you want to edit `TA_VALIDATION_PATH` variable to lead to the path of the validation set and run the module. Outputs should be in `outputs/` folder.

Note that we do expect that each file has the format `something_[time_of_day]_location.jpeg`

The general format should be 
```
validation_set/
├── 01/
│   ├── 01_daytime_cloudy.png
│   ├── 01_daytime_sunny.png
├── 02/
│   ├── 02_night_cloudy.py                y                
│   └── 02_daytime_rainy.py    
```