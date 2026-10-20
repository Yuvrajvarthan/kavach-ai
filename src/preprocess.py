"""Text cleaning, engineered features, template keys and dataset loaders for Kavach AI.

`clean_text` is applied INSIDE the saved sklearn pipeline (via `clean_series`), so the
saved model takes raw text.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# Label ids follow the Sentinel dataset card (Shade63/scam-classification-multiclass).
CLASSES = ["legit", "government_impersonation", "financial_fraud",
           "bank_phishing", "romance_parcel_sextortion"]
LEGIT = "legit"

# --- Regexes (applied to lowercased text) ---
URL_RE = re.compile(
    r"(?:https?://|www\.)\S+"
    r"|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|in|net|org|info|xyz|top|co|ly|me|io|site|online|"
    r"link|click|app|biz|gov|example)\b(?:/\S*)?"
)
# Currency symbol/word (possibly repeated, e.g. "Rs Rs 8,900") + number, or number + rupees/rs.
AMOUNT_RE = re.compile(
    r"(?:(?:₹|\brs\.?|\binr\b|\$|£|€)\s*)+\d[\d,]*(?:\.\d+)?(?:\s*(?:k|lakh|lakhs|crore|cr)\b)?"
    r"|\b\d[\d,]*(?:\.\d+)?\s*(?:rupees|rs\b|inr\b|/-)"
)
# Candidate digit runs (with optional +, spaces, dashes); kept as a phone only if >= 10 digits.
PHONE_CAND_RE = re.compile(r"\+?\d[\d\s\-]{8,}\d")
WS_RE = re.compile(r"\s+")

URGENCY_WORDS = [
    "urgent", "urgently", "immediately", "immediate", "now", "today", "tonight", "within",
    "hours", "minutes", "expire", "expired", "expires", "expiring", "blocked", "block",
    "suspend", "suspended", "suspension", "deactivate", "deactivated", "disconnect",
    "disconnected", "last", "final", "warning", "alert", "asap", "arrest", "warrant",
    "legal", "penalty", "fine", "deadline", "limited", "hurry", "abhi", "turant", "jaldi",
    "aaj", "band", "warna",
]
URGENCY_RE = re.compile(r"\b(?:" + "|".join(URGENCY_WORDS) + r")\b")


def _replace_phones(text: str) -> str:
    def repl(m: re.Match) -> str:
        return " <PHONE> " if sum(ch.isdigit() for ch in m.group()) >= 10 else m.group()
    return PHONE_CAND_RE.sub(repl, text)


def tag_text(text: str) -> str:
    """Lowercase; URLs -> <URL>, amounts -> <AMOUNT>, phones -> <PHONE>; collapse spaces.

    Used for red-flag features and template keys. The model sees `clean_text` instead.
    """
    t = str(text).lower()
    t = URL_RE.sub(" <URL> ", t)
    t = AMOUNT_RE.sub(" <AMOUNT> ", t)
    t = _replace_phones(t)
    return WS_RE.sub(" ", t).strip()


def _phone_spans(s: str):
    return [m.span() for m in PHONE_CAND_RE.finditer(s) if sum(c.isdigit() for c in m.group()) >= 10]


def tag_text_with_map(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Same output as `tag_text`, plus, for every output character, the (start, end) span of
    the raw text it came from. Lets rule matches be shown as the user's original words."""
    raw = str(text)
    low = "".join(c.lower() if len(c.lower()) == 1 else c for c in raw)  # keep 1:1 offsets
    segs = [(low, 0, None)]  # (text, raw_start, tag or None)
    for tag, finder in [("<URL>", lambda s: [m.span() for m in URL_RE.finditer(s)]),
                        ("<AMOUNT>", lambda s: [m.span() for m in AMOUNT_RE.finditer(s)]),
                        ("<PHONE>", _phone_spans)]:
        out = []
        for seg_text, start, seg_tag in segs:
            if seg_tag:
                out.append((seg_text, start, seg_tag))
                continue
            pos = 0
            for a, b in finder(seg_text):
                out.append((seg_text[pos:a], start + pos, None))
                out.append((seg_text[a:b], start + a, tag))
                pos = b
            out.append((seg_text[pos:], start + pos, None))
        segs = out
    chars, spans = [], []
    for seg_text, start, seg_tag in segs:
        if seg_tag:
            span = (start, start + len(seg_text))
            for c in f" {seg_tag} ":
                chars.append(c)
                spans.append(span)
        else:
            for i, c in enumerate(seg_text):
                chars.append(c)
                spans.append((start + i, start + i + 1))
    # collapse whitespace runs to one space and strip, keeping the map aligned
    out_c, out_s = [], []
    for c, sp in zip(chars, spans):
        if c.isspace():
            if not out_c or out_c[-1] == " ":
                continue
            c = " "
        out_c.append(c)
        out_s.append(sp)
    while out_c and out_c[-1] == " ":
        out_c.pop()
        out_s.pop()
    return "".join(out_c), out_s


def raw_span_text(raw: str, spans: list[tuple[int, int]], a: int, b: int) -> str:
    """Original text behind tagged-text characters a..b (from `tag_text_with_map`)."""
    if a >= b:
        return ""
    return str(raw)[min(s for s, _ in spans[a:b]):max(e for _, e in spans[a:b])].strip()


def clean_text(text: str) -> str:
    """Text the model sees: `tag_text` with phone numbers removed entirely.

    In Sentinel, phone numbers occur in 16% of legit rows and 0% of scam rows (a dataset
    quirk), so a <PHONE> token would teach the model "has a phone number -> legit".
    Hinglish words are kept as-is (no stop-word removal, no stemming).
    """
    return WS_RE.sub(" ", tag_text(text).replace("<PHONE>", " ")).strip()


def clean_series(texts) -> list[str]:
    """Vectorised `clean_text`, used as the first pipeline step (must be module-level to pickle)."""
    return [clean_text(t) for t in texts]


# Tokens like <URL> survive as single tokens; otherwise words of 2+ chars.
TOKEN_PATTERN = r"(?u)<\w+>|\b\w\w+\b"


def template_key(text: str) -> str:
    """Normalised text used to group templated messages (digits/URLs/phones/amounts masked)."""
    t = tag_text(text)
    t = re.sub(r"\d+", "#", t)
    t = re.sub(r"[^\w\s<>#]", " ", t)
    return WS_RE.sub(" ", t).strip()


def match_key(text: str) -> str:
    """Loose normalisation for matching the same message across datasets."""
    t = re.sub(r"[^\w\s]", "", str(text).lower())
    return WS_RE.sub(" ", t).strip()


def extract_features(text: str) -> dict:
    """Red-flag features for EDA and the Scan page checklist."""
    low = str(text).lower()
    tagged = tag_text(text)
    return {
        "has_url": "<URL>" in tagged,
        "has_phone": "<PHONE>" in tagged,
        "has_amount": "<AMOUNT>" in tagged,
        "urgency_count": len(URGENCY_RE.findall(low)),
        "length": len(str(text)),
        "n_words": len(str(text).split()),
    }


def add_features(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    feats = pd.DataFrame([extract_features(t) for t in df[text_col]], index=df.index)
    return pd.concat([df, feats], axis=1)


# --- Dataset loaders (files produced by scripts/download_data.py) ---

def load_sentinel(dedupe: bool = True) -> tuple[pd.DataFrame, dict]:
    """Sentinel multiclass with string class names in `label_name`. Returns (df, stats)."""
    df = pd.read_csv(RAW / "sentinel_dataset_multiclass_clean.csv")
    raw_rows = len(df)
    df = df.dropna(subset=["text"])
    if dedupe:
        df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df["label_name"] = df["label"].map(dict(enumerate(CLASSES)))
    stats = {"raw_rows": raw_rows, "duplicates_removed": raw_rows - len(df), "clean_rows": len(df)}
    return df, stats


def _dedupe(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    out = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    return out, len(df) - len(out)


def load_uci() -> tuple[pd.DataFrame, int]:
    """UCI SMS Spam, binary `label` (1 = spam/scam). Returns (deduped df, duplicates dropped)."""
    df = pd.read_csv(RAW / "uci_sms" / "SMSSpamCollection", sep="\t", header=None,
                     names=["label", "text"], quoting=3, encoding="utf-8")
    df["label"] = (df["label"] == "spam").astype(int)
    return _dedupe(df)


def load_hinglish() -> tuple[pd.DataFrame, int]:
    """Hinglish & English scam text, binary `label` (1 = scam). Returns (deduped df, dropped)."""
    text = (RAW / "hinglish_scam.json").read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        df = pd.read_json(io.StringIO(text), lines=True)
    else:
        if isinstance(obj, dict):
            obj = next(v for v in obj.values() if isinstance(v, list))
        df = pd.DataFrame(obj)
    df["label"] = df["label"].astype(int)
    return _dedupe(df[["text", "label"]])


def hinglish_only(hinglish: pd.DataFrame, uci: pd.DataFrame) -> pd.DataFrame:
    """Rows of the Hinglish set whose normalised text does not appear in UCI SMS."""
    uci_keys = set(uci["text"].map(match_key))
    mask = ~hinglish["text"].map(match_key).isin(uci_keys)
    return hinglish[mask].reset_index(drop=True)


if __name__ == "__main__":
    for s in ["URGENT! Your SBI KYC expired. Update at http://sbi-kyc.example/login or call "
              "+91 98765 43210. Pay Rs 1,999 now.",
              "Aapka bijli connection aaj raat kaat diya jayega. Turant ₹500 pay karein 9876543210 pe",
              "Your debit card ending 4521 has been suspended"]:
        print(clean_text(s), "|", template_key(s), "|", extract_features(s))
    df, st = load_sentinel()
    print(st, np.bincount(df["label"]))
