"""Single-message prediction helper used by the app and tools.

    from src.predict import predict
    predict("Your KYC has expired ...")
    -> {"label": "bank_phishing", "probabilities": {...5 classes...},
        "scam_prob": 0.97, "risk": "High"}
    analyse(text) -> predict(text) + {"red_flags": [...], "advisory": "high|caution|none"}
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib

from src.preprocess import CLASSES
from src.red_flags import advisory_level, check_red_flags

MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "model.joblib"

HIGH_RISK = 0.80
MEDIUM_RISK = 0.50


@lru_cache(maxsize=1)
def load_model(path: str | Path = MODEL_PATH):
    return joblib.load(path)


def risk_tier(scam_prob: float) -> str:
    """High >= 0.80, Medium 0.50-0.80, Low < 0.50."""
    if scam_prob >= HIGH_RISK:
        return "High"
    if scam_prob >= MEDIUM_RISK:
        return "Medium"
    return "Low"


def predict(text: str, model=None) -> dict:
    model = model or load_model()
    proba = model.predict_proba([text])[0]
    # model.classes_ are the integer label ids 0..4, in CLASSES order
    probs = {CLASSES[int(c)]: round(float(p), 4) for c, p in zip(model.classes_, proba)}
    scam_prob = round(1.0 - probs["legit"], 4)
    return {
        "label": max(probs, key=probs.get),
        "probabilities": probs,
        "scam_prob": scam_prob,
        "risk": risk_tier(scam_prob),
    }


def analyse(text: str, model=None) -> dict:
    """predict(text) plus rule-based red flags. The rules never change the ML outputs."""
    flags = check_red_flags(text)
    return {**predict(text, model), "red_flags": flags, "advisory": advisory_level(flags)}


if __name__ == "__main__":
    import sys
    print(analyse(" ".join(sys.argv[1:]) or "Your KYC has expired, update now at http://x.example"))
