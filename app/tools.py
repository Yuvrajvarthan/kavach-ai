"""Tools for the Kavach Assistant. The LLM calls these; it never classifies by itself.

classify_message(text)            -> trained ML model + red-flag rules + LIME words
query_dataset(class_name, keyword) -> counts / % / examples from the Sentinel training data
safety_guidance(class_name)        -> vetted advice from src/guidance.py, returned as is
"""
from __future__ import annotations

import re
from functools import lru_cache

from app.scan_core import build_verdict, highlight_words
from src.explain import explain_local
from src.guidance import CLASS_PLAIN, get_guidance, warning_signs_guidance
from src.predict import analyse, load_model
from src.preprocess import CLASSES, load_sentinel


class ToolCallError(ValueError):
    """The LLM asked for an unknown tool or passed bad arguments."""


def classify_message(text: str, model=None) -> dict:
    model = model or load_model()
    res = analyse(text, model)
    expl = explain_local(text, model=model)
    words, _ = highlight_words(res, expl)
    v = build_verdict(res)
    return {
        "classified_by": "trained Linear SVM model (TF-IDF); red flags from hand-written rules",
        "label": res["label"],
        "label_plain": CLASS_PLAIN[res["label"]],
        "verdict_headline": v["headline"],
        "verdict_detail": v["detail"],
        "verdict_notes": v["notes"],
        "risk": res["risk"],
        "scam_prob": res["scam_prob"],
        "probabilities": res["probabilities"],
        "advisory": res["advisory"],
        "red_flags": [{"flag": f["flag"], "label": f["label"], "evidence": f["evidence"]}
                      for f in res["red_flags"]],
        "words_pointing_to_scam": [w for w, x in words if x > 0][:5],
        "words_pointing_to_legit": [w for w, x in words if x < 0][:5],
    }


@lru_cache(maxsize=1)
def _sentinel():
    df, _ = load_sentinel()
    return df


def _resolve_class(name: str | None) -> str | None:
    if not name:
        return None
    key = re.sub(r"[\s\-/]+", "_", name.strip().lower())
    if key in CLASSES:
        return key
    for c in CLASSES:  # e.g. "bank phishing messages", "phishing", "romance"
        if key in c or c in key:
            return c
    raise ToolCallError(f"Unknown class '{name}'. Valid classes: {', '.join(CLASSES)}")


def query_dataset(class_name: str | None = None, keyword: str | None = None) -> dict:
    df = _sentinel()
    cls = _resolve_class(class_name)
    total = len(df)
    sub = df
    if cls:
        sub = sub[sub["label_name"] == cls]
    if keyword:
        sub = sub[sub["text"].str.contains(keyword, case=False, regex=False)]
    out = {
        "dataset": "Sentinel multiclass (the model's training data, synthetic Indian-context SMS)",
        "total_rows": total,
        "filter": {"class": cls, "keyword": keyword},
        "matching_rows": int(len(sub)),
        "pct_of_dataset": round(100 * len(sub) / total, 2),
    }
    if keyword or not cls:
        out["matching_rows_by_class"] = {c: int((sub["label_name"] == c).sum()) for c in CLASSES}
    if cls and keyword:
        n_cls = int((df["label_name"] == cls).sum())
        out["pct_of_class"] = round(100 * len(sub) / n_cls, 2)
    if len(sub):
        ex = sub.sample(min(3, len(sub)), random_state=42)
        out["examples"] = [{"class": r.label_name, "text": r.text[:200]} for r in ex.itertuples()]
    return out


def safety_guidance(class_name: str | None = "general") -> dict:
    if class_name in CLASSES:
        return get_guidance(class_name)
    try:
        cls = _resolve_class(class_name) if class_name and class_name != "general" else None
    except ToolCallError:
        cls = None
    return get_guidance(cls) if cls else warning_signs_guidance()


TOOLS = {"classify_message": classify_message, "query_dataset": query_dataset,
         "safety_guidance": safety_guidance}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "classify_message",
        "description": ("Classify a suspicious SMS/WhatsApp/call message with Kavach's TRAINED "
                        "machine-learning model. Returns the predicted class, probabilities, risk "
                        "level, rule-based red flags and the words that drove the prediction. Call "
                        "this for ANY message the user wants checked."),
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "The exact message text to check."}},
            "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "query_dataset",
        "description": ("Counts, percentages and up to 3 examples from the training dataset "
                        "(14,000 messages). Filter by class and/or keyword."),
        "parameters": {"type": "object", "properties": {
            "class_name": {"type": "string", "enum": CLASSES,
                           "description": "Optional class to filter by."},
            "keyword": {"type": "string", "description": "Optional word or phrase to search for."}}}}},
    {"type": "function", "function": {
        "name": "safety_guidance",
        "description": ("Vetted 'what to do now' steps and the ONLY allowed helpline (1930) and "
                        "website (cybercrime.gov.in). Use the predicted class, or 'general'."),
        "parameters": {"type": "object", "properties": {
            "class_name": {"type": "string", "enum": CLASSES + ["general"]}},
            "required": ["class_name"]}}},
]


def execute_tool(name: str, args: dict) -> dict:
    if name not in TOOLS:
        raise ToolCallError(f"Unknown tool '{name}'")
    if not isinstance(args, dict):
        raise ToolCallError(f"Arguments for {name} must be an object")
    if name == "classify_message" and not str(args.get("text", "")).strip():
        raise ToolCallError("classify_message needs non-empty 'text'")
    allowed = {"classify_message": {"text"}, "query_dataset": {"class_name", "keyword"},
               "safety_guidance": {"class_name"}}[name]
    try:
        return TOOLS[name](**{k: v for k, v in args.items() if k in allowed})
    except TypeError as e:
        raise ToolCallError(f"Bad arguments for {name}: {e}") from e
