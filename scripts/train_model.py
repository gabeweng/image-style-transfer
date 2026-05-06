import sys
import os

# Cell 1
IN_COLAB = 'google.colab' in sys.modules
IS_MACOS = sys.platform == "darwin"

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')

# Cell 2
if IN_COLAB:
    REPO_URL = 'https://github.com/gabeweng/image-style-transfer.git'
    REPO_DIR = '/content/image-style-transfer'

    import os

    if os.path.exists(REPO_DIR):
        %cd {REPO_DIR}
        !git pull
    else:
        %cd /content
        !git clone {REPO_URL} {REPO_DIR}
        %cd {REPO_DIR}

# Cell 3
if IN_COLAB:
    os.system("pip install -q uv")

# xformers is for hardware acceleration with Nvidia drivers, something macos doesn't support
if not IS_MACOS:
    os.system("uv pip install xformers")
os.system("uv pip install pillow pandas torch torchvision diffusers==0.27.2 'transformers>=4.38.0,<5' 'huggingface-hub<0.26' accelerate peft datasets safetensors wandb tqdm")

# Cell 4
if IN_COLAB:
    BASE = '/content/drive/My Drive/CIS_5190_group_project'
else:
    # We're likely in the notebooks folder
    BASE = '..'

# Can change to 'manifest.csv' to 'audit_decisions.csv' to work on human approved images
MANIFEST_CSV = f'{BASE}/manifest.csv'
FILTERED_DIR = f'{BASE}/filtered_aligned'
HF_DATASET_DIR = f'{BASE}/hf_dataset'
HF_CONTROLNET_DIR = f'{BASE}/data/hf_dataset_controlnet'
CHECKPOINT_DIR = f'{BASE}/checkpoints'

print('Manifest:', MANIFEST_CSV)
print('Filtered images:', FILTERED_DIR)
print('HF dataset:', HF_DATASET_DIR)
print('Optional ControlNet dataset:', HF_CONTROLNET_DIR)
print('Checkpoints:', CHECKPOINT_DIR)

# Cell 5
import json
import os
import pandas as pd

assert os.path.exists(MANIFEST_CSV), f'Missing manifest: {MANIFEST_CSV}'
assert os.path.isdir(FILTERED_DIR), f'Missing filtered image directory: {FILTERED_DIR}'
assert os.path.exists(f'{HF_DATASET_DIR}/metadata.jsonl'), f'Missing HF metadata: {HF_DATASET_DIR}/metadata.jsonl'

manifest_df = pd.read_csv(MANIFEST_CSV)
kept_df = manifest_df[manifest_df['status'] == 'kept'].copy() if 'status' in manifest_df.columns else manifest_df.copy()

print(f'Manifest rows: {len(manifest_df)}')
print(f'Kept final images: {len(kept_df)}')
if 'status' in manifest_df.columns:
    print('Status counts:', manifest_df['status'].value_counts().to_dict())
if len(kept_df):
    print('Train/val split:', kept_df['split'].value_counts().to_dict())
    print(kept_df.head())
    # display(kept_df.head())

missing = [p for p in kept_df['file_name'].head(20) if not os.path.exists(os.path.join(BASE, p))]
assert not missing, f'Some kept manifest files were not found: {missing[:5]}'

# Might need to modify this to work with manual auditing (if enough time)
with open(f'{HF_DATASET_DIR}/metadata.jsonl') as f:
    metadata_rows = sum(1 for _ in f)
print(f'HF metadata rows: {metadata_rows}')
print('Preprocess/alignment artifacts are ready for training.')


# Cell 6
import torch

print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))


# Cell 7
USE_WANDB = False
WANDB_PROJECT = 'image-style-transfer'

if USE_WANDB:
    import os
    os.environ['WANDB_PROJECT'] = WANDB_PROJECT
    os.environ['WANDB_LOG_MODEL'] = 'checkpoint'
    !wandb login
    REPORT_TO_ARG = '--report_to=wandb'
else:
    REPORT_TO_ARG = ''

print('W&B enabled:', USE_WANDB)


# Cell 8
if IN_COLAB:
    DIFFUSERS_DIR = '/content/diffusers'
else:
    DIFFUSERS_DIR = '../diffusers'

if os.path.exists(DIFFUSERS_DIR):
    os.system(f"cd {DIFFUSERS_DIR}")
    os.system("git fetch --tags")
    os.system("git checkout v0.27.2")
else:
    os.system("cd /content")
    os.system("git clone --branch v0.27.2 --depth 1 https://github.com/huggingface/diffusers.git {DIFFUSERS_DIR}")

os.system(f"cd {REPO_DIR}")

# Cell 9