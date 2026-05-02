"""
audit_app.py  —  Streamlit alignment verification tool

Run locally:
    streamlit run scripts/audit_app.py -- \
        --aligned_csv  /path/to/aligned_labels.csv \
        --aligned_dir  /path/to/aligned \
        --eval_csv     /path/to/lpips_eval_set.csv

The app lets you scrub through anchor/warped pairs per location,
blend them with a slider to spot geometry errors, and approve or
reject each pair for the strict LPIPS evaluation set.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# CLI args (Streamlit passes everything after "--" to the script)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--aligned_csv", default="aligned_labels.csv")
parser.add_argument("--aligned_dir", default="aligned")
parser.add_argument("--eval_csv", default="lpips_eval_set.csv")
args, _ = parser.parse_known_args()

ALIGNED_CSV = args.aligned_csv
ALIGNED_DIR = args.aligned_dir
EVAL_CSV = args.eval_csv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data
def load_aligned_df(path):
    return pd.read_csv(path)


def load_img(fname):
    p = os.path.join(ALIGNED_DIR, fname)
    if not os.path.exists(p):
        return None
    return Image.open(p).convert("RGB")


def blend(img_a: Image.Image, img_b: Image.Image, alpha: float) -> Image.Image:
    a = np.array(img_a, dtype=np.float32)
    b = np.array(img_b, dtype=np.float32)
    return Image.fromarray(((1 - alpha) * a + alpha * b).astype(np.uint8))


def load_eval_set(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=["location", "anchor_file", "target_file",
                                  "target_tod", "target_weather", "warped_path", "ssim_score"])


def save_eval_set(df, path):
    df.to_csv(path, index=False)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Alignment Audit", layout="wide")
st.title("Alignment Audit — LPIPS Pair Curation")

if not os.path.exists(ALIGNED_CSV):
    st.error(f"aligned_labels.csv not found at: {ALIGNED_CSV}\n"
             "Run `python scripts/align_images.py` first.")
    st.stop()

df = load_aligned_df(ALIGNED_CSV)
eval_df = load_eval_set(EVAL_CSV)

# Sidebar — location selector
locations = sorted(df["location"].unique())
location = st.sidebar.selectbox("Location", locations)

loc_df = df[df["location"] == location].reset_index(drop=True)
pair_labels = [
    f"{r['target_tod']} / {r['target_weather']}  (SSIM={r['ssim_score']:.3f})"
    for _, r in loc_df.iterrows()
]

pair_idx = st.sidebar.selectbox("Pair", range(len(pair_labels)), format_func=lambda i: pair_labels[i])
row = loc_df.iloc[pair_idx]

already_approved = (
    not eval_df.empty
    and ((eval_df["location"] == row["location"]) & (eval_df["target_file"] == row["target_file"])).any()
)

# Main panel
col_l, col_r = st.columns(2)
anchor_img = load_img(row["anchor_file"])
warped_img = load_img(row["warped_path"])

if anchor_img is None or warped_img is None:
    st.error("Could not load images — check --aligned_dir path.")
    st.stop()

alpha = st.slider("Blend  (0 = anchor, 1 = warped target)", 0.0, 1.0, 0.5, 0.05)
blended = blend(anchor_img, warped_img, alpha)

with col_l:
    st.image(anchor_img, caption=f"Anchor  ({row['anchor_tod']} / {row['anchor_weather']})", use_container_width=True)
with col_r:
    st.image(warped_img, caption=f"Warped target  ({row['target_tod']} / {row['target_weather']})", use_container_width=True)

st.image(blended, caption="Blended preview", use_container_width=True)

st.markdown(f"**SSIM:** {row['ssim_score']:.4f} &nbsp;|&nbsp; **Keypoint matches:** {int(row.get('n_matches', 0))}")

if already_approved:
    st.success("This pair is already in the LPIPS eval set.")

c1, c2 = st.columns(2)
with c1:
    if st.button("✅  Approve Pair  →  LPIPS eval set", disabled=already_approved):
        new_row = row.to_dict()
        eval_df = pd.concat([eval_df, pd.DataFrame([new_row])], ignore_index=True)
        save_eval_set(eval_df, EVAL_CSV)
        st.success(f"Saved to {EVAL_CSV}")
        st.cache_data.clear()

with c2:
    if st.button("🗑️  Reject  (train only)"):
        if already_approved:
            eval_df = eval_df[
                ~((eval_df["location"] == row["location"]) & (eval_df["target_file"] == row["target_file"]))
            ]
            save_eval_set(eval_df, EVAL_CSV)
            st.warning("Removed from LPIPS eval set.")
            st.cache_data.clear()
        else:
            st.info("Not in eval set — nothing to remove.")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Eval set size:** {len(eval_df)} pairs")
