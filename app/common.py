"""Shared Streamlit helpers: paths, cached loaders, small UI pieces."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import streamlit as st  # noqa: E402

from src.guidance import HELPLINE, PORTAL  # noqa: E402
from src.predict import MODEL_PATH  # noqa: E402
from src.preprocess import CLASSES  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "reports" / "figures"
DEMO_DIR = ROOT / "demo_samples"

# Same fixed class colours as the report figures (src/evaluate.py CLASS_COLORS)
CLASS_COLORS = dict(zip(CLASSES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
SPLIT_COLORS = {"Random split": "#2a78d6", "Template-grouped split": "#eb6834"}


def page_setup(title: str, icon: str):
    st.set_page_config(page_title=f"{title} · Kavach AI", page_icon=icon, layout="wide")
    with st.sidebar:
        st.markdown("### 🛡️ Kavach AI")
        st.caption("Scam & fraud message detector with explainable ML")
        st.markdown(f"**Lost money to fraud?**  \nCall **{HELPLINE}** or report at "
                    f"[cybercrime.gov.in]({PORTAL})")


@st.cache_resource(show_spinner="Loading the scam classifier…")
def get_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def get_metrics() -> dict:
    return json.loads((ARTIFACTS / "metrics.json").read_text())


@st.cache_data
def get_confusion() -> dict:
    return json.loads((ARTIFACTS / "confusion_matrices.json").read_text())


@st.cache_data
def get_demo_messages() -> list[dict]:
    return json.loads((DEMO_DIR / "messages.json").read_text())


def demo_files(pattern: str) -> list[Path]:
    return sorted(DEMO_DIR.glob(pattern))


def leaderboard(metrics: dict, compact: bool = False) -> "pd.DataFrame":  # noqa: F821
    import pandas as pd
    rows = []
    for name in metrics["models"]:
        r, g = metrics["models"][name], metrics["models_grouped"][name]
        rows.append({"Model": name, "Macro-F1 (grouped)": g["macro_f1"],
                     "Macro-F1 (random)": r["macro_f1"], "Accuracy (grouped)": g["accuracy"],
                     "Accuracy (random)": r["accuracy"], "ROC-AUC (grouped)": g["roc_auc_ovr"],
                     "Train time (s)": r["train_time_s"]})
    df = pd.DataFrame(rows).sort_values("Macro-F1 (grouped)", ascending=False)
    df.insert(0, "Best", ["⭐" if m == metrics["best_model"] else "" for m in df["Model"]])
    if compact:
        df = df[["Best", "Model", "Macro-F1 (grouped)", "Macro-F1 (random)"]]
    return df.reset_index(drop=True)
