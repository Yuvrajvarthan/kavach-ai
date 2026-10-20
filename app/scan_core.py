"""Scan pipeline logic shared by the Scan page and offline checks (no Streamlit here).

run_scan(text) -> ML result (analyse) + LIME words + verdict card + signals + guidance.
"""
from __future__ import annotations

import html
import re

from src.explain import explain_local
from src.guidance import CLASS_PLAIN, get_guidance, warning_signs_guidance
from src.predict import analyse
from src.preprocess import AMOUNT_RE, PHONE_CAND_RE, URGENCY_RE, URL_RE

# Verdict card colours: (accent, background, text) - text is always dark for contrast
CARD_STYLES = {
    "scam_high": ("#c62828", "#fde8e8", "#3b0a0a"),
    "scam_medium": ("#e65100", "#fff0e0", "#3d1a00"),
    "warning": ("#b26a00", "#fff4d6", "#3a2600"),
    "caution": ("#c99a2e", "#fffaeb", "#3a2e10"),
    "clear": ("#2e7d32", "#e8f5e9", "#0f2e12"),
}


def build_verdict(res: dict) -> dict:
    """Verdict card from analyse() output, following the fixed table:
    ML scam -> red/orange; ML legit + advisory high -> amber; + caution -> light amber;
    + none -> green ("No scam pattern detected", never "Safe"). `notes` are extra lines shown
    under the card when an ML scam verdict is uncertain or has no classic warning signs."""
    label, flags, advisory = res["label"], res["red_flags"], res["advisory"]
    if label != "legit":
        style = "scam_high" if res["risk"] == "High" else "scam_medium"
        headline = f"Likely scam: {CLASS_PLAIN[label]}"
        detail = f"Risk level: {res['risk']}"
    elif advisory == "high":
        style = "warning"
        headline = f"Be careful: {len(flags)} warning signs found"
        detail = ("Our model didn't match this to a scam type it was trained on, "
                  "but it shows classic scam signs.")
    elif advisory == "caution":
        style = "caution"
        headline = "No scam pattern matched, but check: " + "; ".join(f["label"].lower() for f in flags)
        detail = ""
    else:
        style = "clear"
        headline = "No scam pattern detected"
        detail = ""
    notes = []
    if label != "legit":
        top_p = res["probabilities"][label]
        if top_p < 0.50:
            notes.append(f"The model isn't sure which scam type this is (top match {top_p:.0%}).")
        if advisory == "none":
            found = ("No classic warning signs were found." if not flags else
                     f"Only one classic warning sign was found ({flags[0]['label'].lower()}).")
            notes.append(f"{found} If you know the sender, confirm with them on a number you "
                         "already have.")
    accent, bg, fg = CARD_STYLES[style]
    return {"style": style, "headline": headline, "detail": detail, "notes": notes,
            "footer": f"Model: {label} · scam probability {res['scam_prob']:.0%}",
            "accent": accent, "bg": bg, "fg": fg}


def _unique(items) -> list[str]:
    out = []
    for i in items:
        i = i.strip()
        if i and i not in out:
            out.append(i)
    return out


def basic_signals(text: str) -> list[dict]:
    """URL / phone / amount / urgency checklist with the original matched text as evidence."""
    raw = str(text)
    low = "".join(c.lower() if len(c.lower()) == 1 else c for c in raw)  # 1:1 offsets with raw

    def found(regex, keep=lambda s: True):
        return _unique(raw[m.start():m.end()] for m in regex.finditer(low) if keep(m.group()))

    return [
        {"id": "url", "label": "Contains a link", "evidence": found(URL_RE)},
        {"id": "phone", "label": "Contains a phone number",
         "evidence": found(PHONE_CAND_RE, lambda s: sum(c.isdigit() for c in s) >= 10)},
        {"id": "amount", "label": "Mentions a money amount", "evidence": found(AMOUNT_RE)},
        {"id": "urgency", "label": "Uses urgency words", "evidence": found(URGENCY_RE)},
    ]


def highlight_words(res: dict, expl: dict) -> tuple[list[tuple[str, float]], str]:
    """Words with weights relative to a SCAM class (+ = toward scam, - = toward legit),
    and the name of that class. For a legit prediction LIME's scam_words are used."""
    if res["label"] == "legit":
        return expl["scam_words"], expl["scam_class"]
    return expl["words"], expl["label"]


def highlight_html(text: str, words: list[tuple[str, float]]) -> str:
    """Message as HTML with LIME words marked: red = toward scam, green = toward legit."""
    weights = {w: v for w, v in words}
    top = max((abs(v) for v in weights.values()), default=0) or 1.0
    out, pos = [], 0
    for m in re.finditer(r"\w+", text):
        out.append(html.escape(text[pos:m.start()]))
        tok = m.group()
        if tok in weights and weights[tok] != 0:
            v = weights[tok]
            alpha = 0.25 + 0.6 * abs(v) / top
            rgb = "220,38,38" if v > 0 else "22,163,74"
            out.append(f'<mark title="{v:+.3f}" style="background:rgba({rgb},{alpha:.2f});'
                       f'color:inherit;padding:0 2px;border-radius:3px">{html.escape(tok)}</mark>')
        else:
            out.append(html.escape(tok))
        pos = m.end()
    out.append(html.escape(text[pos:]))
    return "".join(out)


def pick_guidance(res: dict) -> dict:
    if res["label"] != "legit":
        return get_guidance(res["label"])
    if res["advisory"] in ("high", "caution"):
        return warning_signs_guidance()
    return get_guidance("legit")


def run_scan(text: str, model=None) -> dict:
    res = analyse(text, model)
    expl = explain_local(text, model=model)
    words, toward = highlight_words(res, expl)
    return {
        "text": text,
        "result": res,
        "explanation": expl,
        "highlight_words": words,
        "highlight_class": toward,
        "verdict": build_verdict(res),
        "signals": basic_signals(text),
        "guidance": pick_guidance(res),
    }
