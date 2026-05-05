"""
audit_app_copy.py - Streamlit HF dataset image audit tool

Run locally:
    streamlit run scripts/audit_app_copy.py -- \
        --dataset_dir hf_dataset \
        --manifest_csv hf_dataset/metadata.csv \
        --output_csv hf_dataset/audit_decisions.csv

This app lets you go through every image in a Hugging Face ImageFolder
dataset by location, labeling each location/time_of_day/weather image as
approved or rejected. We just want another pair of eyes to make sure that nothing
too funky is going on with the images.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError


AUDIT_COLUMNS = [
    "file_name",
    "location",
    "time_of_day",
    "weather",
    "decision",
]

REQUIRED_COLUMNS = {"file_name", "location", "time_of_day", "weather"}


# ---------------------------------------------------------------------------
# CLI args (Streamlit passes everything after "--" to the script)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--dataset_dir", default="hf_dataset")
parser.add_argument("--manifest_csv", default=None)
parser.add_argument("--output_csv", default=None)
args, _ = parser.parse_known_args()

DATASET_DIR = os.path.abspath(args.dataset_dir)


def default_manifest_path(dataset_dir: str) -> str:
    candidates = [
        os.path.join(dataset_dir, "manifest.csv"),
        os.path.join(dataset_dir, "metadata.csv"),
        os.path.join(dataset_dir, "metadata.jsonl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


MANIFEST_PATH = os.path.abspath(args.manifest_csv or default_manifest_path(DATASET_DIR))
AUDIT_PATH = os.path.abspath(args.output_csv or os.path.join(DATASET_DIR, "audit_decisions.csv"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data
def load_manifest(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    else:
        df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {', '.join(sorted(missing))}")

    df = df.copy()
    for col in REQUIRED_COLUMNS:
        df[col] = df[col].fillna("").astype(str)

    if "status" in df.columns:
        kept = df[df["status"].fillna("").astype(str).str.lower().eq("kept")]
        if not kept.empty:
            df = kept

    return df.sort_values(["location", "time_of_day", "weather", "file_name"]).reset_index(drop=True)


@st.cache_data
def load_audit(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=AUDIT_COLUMNS)

    df = pd.read_csv(path)
    for col in AUDIT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[AUDIT_COLUMNS]


def image_path(row: pd.Series) -> str:
    file_name = str(row["file_name"])
    if os.path.isabs(file_name):
        return file_name
    return os.path.join(DATASET_DIR, file_name)


def open_image(row: pd.Series) -> Image.Image | None:
    path = image_path(row)
    if not os.path.exists(path):
        return None
    try:
        return Image.open(path).convert("RGB")
    except (OSError, UnidentifiedImageError):
        return None


def image_key(row: pd.Series) -> str:
    return str(row["file_name"])


def current_decision(audit_df: pd.DataFrame, row: pd.Series) -> str:
    key = image_key(row)
    matches = audit_df[audit_df["file_name"].astype(str).eq(key)]
    if matches.empty:
        return "unreviewed"
    return str(matches.iloc[-1]["decision"] or "unreviewed")


def save_decision(audit_df: pd.DataFrame, row: pd.Series, decision: str, path: str) -> None:
    key = image_key(row)
    record = {
        "file_name": key,
        "location": row["location"],
        "time_of_day": row["time_of_day"],
        "weather": row["weather"],
        "decision": decision,
    }

    kept = audit_df[~audit_df["file_name"].astype(str).eq(key)].copy()
    out = pd.concat([kept, pd.DataFrame([record])], ignore_index=True)
    out = out.sort_values(["location", "time_of_day", "weather", "file_name"])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out.to_csv(path, index=False)
    st.cache_data.clear()


def condition_label(row: pd.Series) -> str:
    return f"{row['time_of_day']} / {row['weather']}"


def image_label(row: pd.Series, audit_df: pd.DataFrame) -> str:
    status = current_decision(audit_df, row)
    return f"{condition_label(row)} - {Path(str(row['file_name'])).name} [{status}]"


def status_counts(audit_df: pd.DataFrame, manifest_df: pd.DataFrame) -> dict[str, int]:
    reviewed = audit_df.drop_duplicates("file_name", keep="last")
    approved = int(reviewed["decision"].eq("approved").sum()) if not reviewed.empty else 0
    rejected = int(reviewed["decision"].eq("rejected").sum()) if not reviewed.empty else 0
    return {
        "approved": approved,
        "rejected": rejected,
        "unreviewed": max(len(manifest_df) - approved - rejected, 0),
    }


def rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:  # pragma: no cover - old Streamlit fallback
        st.experimental_rerun()


def reset_image_idx() -> None:
    st.session_state.image_idx = 0


def sync_image_idx() -> None:
    st.session_state.image_idx = st.session_state.image_choice


def is_first_image(location: str, locations: list[str]) -> bool:
    return locations.index(location) == 0 and st.session_state.image_idx == 0


def is_last_image(location: str, locations: list[str], location_counts: dict[str, int]) -> bool:
    return (
        locations.index(location) == len(locations) - 1
        and st.session_state.image_idx >= location_counts[location] - 1
    )


def move_image(delta: int, locations: list[str], location_counts: dict[str, int]) -> None:
    location = st.session_state.location
    location_idx = locations.index(location)
    max_idx = location_counts[location] - 1

    if delta > 0 and st.session_state.image_idx >= max_idx:
        if location_idx < len(locations) - 1:
            st.session_state.location = locations[location_idx + 1]
            st.session_state.image_idx = 0
        return

    if delta < 0 and st.session_state.image_idx <= 0:
        if location_idx > 0:
            prev_location = locations[location_idx - 1]
            st.session_state.location = prev_location
            st.session_state.image_idx = location_counts[prev_location] - 1
        return

    st.session_state.image_idx = min(max(st.session_state.image_idx + delta, 0), max_idx)


def audit_current_image(
    row: pd.Series,
    decision: str,
    audit_path: str,
    advance: bool,
    locations: list[str],
    location_counts: dict[str, int],
) -> None:
    audit_df = load_audit(audit_path)
    save_decision(audit_df, row, decision, audit_path)
    if advance:
        move_image(1, locations, location_counts)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
st.set_page_config(page_title="HF Dataset Audit", layout="wide")
st.title("HF Dataset Audit")

if not os.path.isdir(DATASET_DIR):
    st.error(f"HF dataset directory not found: {DATASET_DIR}")
    st.stop()

if not os.path.exists(MANIFEST_PATH):
    st.error(f"Manifest not found: {MANIFEST_PATH}")
    st.stop()

try:
    manifest_df = load_manifest(MANIFEST_PATH)
except Exception as exc:
    st.error(f"Could not load manifest: {exc}")
    st.stop()

audit_df = load_audit(AUDIT_PATH)

locations = sorted(manifest_df["location"].unique())
if not locations:
    st.error("Manifest has no auditable rows.")
    st.stop()
location_counts = manifest_df.groupby("location").size().astype(int).to_dict()


with st.sidebar:
    st.markdown("### Dataset")
    st.caption(DATASET_DIR)
    st.markdown("### Manifest")
    st.caption(MANIFEST_PATH)
    st.markdown("### Decisions")
    st.caption(AUDIT_PATH)

    selected_location = st.selectbox("Location", locations, key="location", on_change=reset_image_idx)
    loc_df = manifest_df[manifest_df["location"].eq(selected_location)].reset_index(drop=True)

    if "image_idx" not in st.session_state:
        st.session_state.image_idx = 0
    st.session_state.image_idx = min(st.session_state.image_idx, max(len(loc_df) - 1, 0))
    st.session_state.image_choice = st.session_state.image_idx

    selected_idx = st.selectbox(
        "Image",
        range(len(loc_df)),
        index=st.session_state.image_idx,
        format_func=lambda idx: image_label(loc_df.iloc[idx], audit_df),
        key="image_choice",
        on_change=sync_image_idx,
    )
    st.session_state.image_idx = selected_idx

    advance_after_decision = st.checkbox("Advance after decision", value=True)

    counts = status_counts(audit_df, manifest_df)
    st.markdown("---")
    st.metric("Approved", counts["approved"])
    st.metric("Rejected", counts["rejected"])
    st.metric("Unreviewed", counts["unreviewed"])

row = loc_df.iloc[st.session_state.image_idx]
img = open_image(row)
path = image_path(row)
decision = current_decision(audit_df, row)

header_l, header_r = st.columns([3, 1])
with header_l:
    st.subheader(f"{row['location']} - {condition_label(row)}")
    st.caption(str(row["file_name"]))
with header_r:
    st.metric("Image", f"{st.session_state.image_idx + 1} / {len(loc_df)}")
    st.write(f"Current: **{decision}**")

if img is None:
    st.error(f"Could not load image: {path}")
else:
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        st.image(img, width="stretch")

nav_l, nav_m, nav_r = st.columns([1, 2, 1])
with nav_l:
    st.button(
        "Previous",
        width="stretch",
        disabled=is_first_image(selected_location, locations),
        on_click=move_image,
        args=(-1, locations, location_counts),
    )
with nav_r:
    st.button(
        "Next",
        width="stretch",
        disabled=is_last_image(selected_location, locations, location_counts),
        on_click=move_image,
        args=(1, locations, location_counts),
    )

approve_col, reject_col = st.columns(2)
with approve_col:
    st.button(
        "Approve Image",
        type="primary",
        width="stretch",
        on_click=audit_current_image,
        args=(row, "approved", AUDIT_PATH, advance_after_decision, locations, location_counts),
    )

with reject_col:
    st.button(
        "Reject Image",
        width="stretch",
        on_click=audit_current_image,
        args=(row, "rejected", AUDIT_PATH, advance_after_decision, locations, location_counts),
    )

details = {
    col: row[col]
    for col in ["location", "time_of_day", "weather", "split", "text", "caption", "status"]
    if col in row.index
}
with st.expander("Manifest row"):
    st.json(details)
