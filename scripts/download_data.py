"""Download the 3 Kavach AI datasets into data/raw/ and print a quick summary.

Usage:  python scripts/download_data.py [--force]

Datasets (all free, no login):
  1. Sentinel multiclass (primary train/test)     -> data/raw/sentinel_dataset_multiclass_clean.csv
  2. Hinglish & English scam text (OOD, CC BY 4.0) -> data/raw/hinglish_scam.json
  3. UCI SMS Spam Collection (OOD, CC BY 4.0)      -> data/raw/uci_sms/SMSSpamCollection
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import requests
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

SENTINEL_REPO = "Shade63/scam-classification-multiclass"
SENTINEL_FILE = "sentinel_dataset_multiclass_clean.csv"
HINGLISH_REPO = "bolewara/hinglish-scam-text-dataset"
HINGLISH_FILE = "data.json"
UCI_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

SENTINEL_PATH = RAW / SENTINEL_FILE
HINGLISH_PATH = RAW / "hinglish_scam.json"
UCI_DIR = RAW / "uci_sms"
UCI_PATH = UCI_DIR / "SMSSpamCollection"


def _hf_download(repo_id: str, filename: str, dest: Path, force: bool) -> Path:
    if dest.exists() and not force:
        print(f"  [skip] {dest.relative_to(ROOT)} already exists")
        return dest
    last_err = None
    # The Hinglish set is published both as a dataset repo and a model repo.
    for repo_type in ("dataset", "model"):
        try:
            cached = hf_hub_download(repo_id, filename, repo_type=repo_type)
            shutil.copyfile(cached, dest)
            print(f"  [ok]   {repo_type}:{repo_id}/{filename} -> {dest.relative_to(ROOT)}")
            return dest
        except Exception as e:  # noqa: BLE001 - try the next repo type
            last_err = e
    raise RuntimeError(f"Could not download {repo_id}/{filename}: {last_err}")


def download_uci(force: bool) -> Path:
    if UCI_PATH.exists() and not force:
        print(f"  [skip] {UCI_PATH.relative_to(ROOT)} already exists")
        return UCI_PATH
    UCI_DIR.mkdir(parents=True, exist_ok=True)
    resp = requests.get(UCI_URL, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for member in zf.namelist():
            name = Path(member).name
            if name in ("SMSSpamCollection", "readme"):
                (UCI_DIR / name).write_bytes(zf.read(member))
    print(f"  [ok]   {UCI_URL} -> {UCI_PATH.relative_to(ROOT)}")
    return UCI_PATH


def load_sentinel() -> pd.DataFrame:
    return pd.read_csv(SENTINEL_PATH)


def load_hinglish() -> pd.DataFrame:
    text = HINGLISH_PATH.read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:  # JSON Lines
        return pd.read_json(io.StringIO(text), lines=True)
    if isinstance(obj, dict):
        # e.g. {"data": [...]} or {"train": [...]}
        lists = [v for v in obj.values() if isinstance(v, list)]
        obj = lists[0] if lists else [obj]
    return pd.DataFrame(obj)


def load_uci() -> pd.DataFrame:
    return pd.read_csv(UCI_PATH, sep="\t", header=None, names=["label", "text"],
                       quoting=3, encoding="utf-8")


def summarise(name: str, df: pd.DataFrame) -> None:
    print(f"\n=== {name} ===")
    print(f"shape:   {df.shape}")
    print(f"columns: {list(df.columns)}")
    print(f"dtypes:  {dict(df.dtypes.astype(str))}")
    # Label-like columns = low cardinality; don't assume their names.
    label_cols = [c for c in df.columns
                  if df[c].dtype != float and df[c].nunique(dropna=False) <= 20]
    for c in label_cols:
        print(f"\nlabel counts — '{c}':")
        print(df[c].value_counts(dropna=False).to_string())
    print("\nsample rows:")
    with pd.option_context("display.max_colwidth", 80, "display.width", 200):
        print(df.head(3).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download even if files exist")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    print("Downloading datasets into data/raw/ ...")
    _hf_download(SENTINEL_REPO, SENTINEL_FILE, SENTINEL_PATH, args.force)
    _hf_download(HINGLISH_REPO, HINGLISH_FILE, HINGLISH_PATH, args.force)
    download_uci(args.force)

    summarise("Sentinel multiclass (primary)", load_sentinel())
    summarise("Hinglish & English scam text (OOD)", load_hinglish())
    summarise("UCI SMS Spam Collection (OOD)", load_uci())


if __name__ == "__main__":
    main()
