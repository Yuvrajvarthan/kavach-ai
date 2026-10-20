"""Rule-based red flags, separate from the ML model.

These rules never change the model's label, probabilities, scam_prob or risk. They catch
common scam tactics (threats, urgency, payment/OTP requests, impersonation) in English and
Hindi/Hinglish, including scam types absent from the training data (e.g. utility
disconnection, "new number, send money"). Written from general scam-awareness guidance
(RBI / cybercrime.gov.in / bank advisories), not fitted to any evaluation set.

    check_red_flags(text) -> [{"flag": id, "label": short English text, "evidence": [words]}]
"""
from __future__ import annotations

import re

from src.preprocess import raw_span_text, tag_text_with_map

_I = re.IGNORECASE

# (flag id, label, patterns). Patterns run on the tagged text (lowercase, with <URL>,
# <AMOUNT>, <PHONE> tags); evidence is reported as the user's original words.
RULES: list[tuple[str, str, list[str]]] = [
    ("threat_disconnect_block", "Threatens to disconnect, block or suspend something", [
        r"\bdisconnect(?:ed|ion)?\b", r"\bcut off\b", r"\b(?:will|would|be|been|get) cut\b",
        r"\bcut ho(?: jayega| jaayega| jaega)?\b", r"\bblock(?:ed|ing)?\b",
        r"\bsuspen(?:d|ded|sion)\b", r"\bdeactivat\w*\b", r"\bterminat\w*\b",
        r"\bkaat\b", r"\bkat (?:diya|di|jayega|jaayega)\b",
        r"\bband (?:ho|kar|kiya|kar diya)\w*\b",
    ]),
    ("urgency", "Pushes you to act immediately", [
        r"\burgent(?:ly)?\b", r"\bimmediate(?:ly)?\b", r"\btoday\b", r"\btonight\b", r"\basap\b",
        r"\bwithin \d+\s*(?:min(?:ute)?s?|hours?|hrs?)\b", r"\blast (?:chance|date|warning)\b",
        r"\bfinal (?:notice|warning|reminder)\b", r"\bturant\b", r"\babhi\b", r"\bjaldi\b",
        r"\baaj raat\b", r"\baaj hi\b",
    ]),
    ("payment_request", "Asks you to pay or transfer money", [
        r"\bpay\b", r"\bpay karein\b", r"\btransfer\b", r"\bg ?pay\b", r"\bgoogle pay\b",
        r"\bphone ?pe\b", r"\bpaytm\b", r"\bupi\b", r"\bkar de\b", r"\bkar do\b",
        r"\bbhej\w*\b", r"\bsend (?:me |us )?(?:money|<AMOUNT>|the amount|payment)\b",
        r"\b(?:registration|processing|clearance|activation) (?:fee|charges?)\b",
    ]),
    ("share_otp_pin", "Mentions OTP, PIN, CVV or password", [
        r"\botp\b", r"\bm?pin\b", r"\bcvv\b", r"\bpassword\b",
    ]),
    ("call_this_number", "Asks you to call or message a number", [
        r"\b(?:call|contact|dial|whatsapp|message|sms|ring)\b[^.!?\n]{0,40}<PHONE>",
        r"\bpress \d\b",
        r"\bis (?:number|no\.?) (?:par|pe|pr) (?:call|baat|contact|whatsapp)",
        r"\bcall (?:this|the above|below|given) number\b",
    ]),
    ("link_present", "Contains a link", [r"<URL>"]),
    ("secrecy", "Tells you to keep it secret", [
        r"\b(?:do not|don't|dont|never) (?:tell|inform|share this with) (?:anyone|anybody|your family|family)\b",
        r"\b(?:do not|don't|dont) disconnect\b",
        r"\bkeep (?:this|it) (?:secret|confidential)\b",
        r"\bkisi (?:ko|se) (?:bhi )?(?:mat|na|nahi) (?:batana|batao|bataye\w*|bolna)\b",
    ]),
    ("new_number_impersonation", "Claims to be someone you know on a new number", [
        r"\bnew (?:number|no\.?)\b", r"\bnaya (?:number|no\.?)\b",
        r"\bchanged my (?:number|phone)\b", r"\bpurana (?:number |phone |no\.? )?kharab\b",
        r"\bmera number badal\w*\b",
    ]),
    ("authority_impersonation", "Claims to be police, a government body, court or official", [
        r"\bpolice\b", r"\bcbi\b", r"\bcustoms\b", r"\bofficer\b", r"\brbi\b",
        r"\bcourt\b", r"\bnarcotics\b", r"\bncb\b", r"\bincome tax\b", r"\btrai\b",
        r"\bcyber ?(?:crime|cell)\b", r"\benforcement directorate\b",
        r"\belectricity (?:board|department|office|officer)\b",
        r"\bbijli (?:vibhag|board|vibhaag|daftar)\b", r"\barrest\b", r"\bwarrant\b", r"\bfir\b",
    ]),
]

# OTP/PIN mentioned only inside a "do not share" warning is the legit pattern.
_SAFE_OTP_RE = re.compile(
    r"\b(?:do not|don't|dont|never|not to)\s+(?:share|disclose|tell)\b|\bkisi (?:ko|se) (?:bhi )?(?:mat|na|nahi) (?:share|bata)", _I)

_COMPILED = [(fid, label, [re.compile(p, _I) for p in pats]) for fid, label, pats in RULES]

ADVISORY_HIGH = 3
ADVISORY_CAUTION = 2


def check_red_flags(text: str) -> list[dict]:
    t, spans = tag_text_with_map(text)
    flags = []
    for fid, label, pats in _COMPILED:
        evidence = []
        for p in pats:
            for m in p.finditer(t):
                w = raw_span_text(text, spans, *m.span())
                if w and w.lower() not in (e.lower() for e in evidence):
                    evidence.append(w)
        if not evidence:
            continue
        if fid == "share_otp_pin" and _SAFE_OTP_RE.search(t):
            continue
        flags.append({"flag": fid, "label": label, "evidence": evidence})
    return flags


def advisory_level(flags: list[dict]) -> str:
    """"high" if 3+ flags, "caution" if 2, else "none"."""
    n = len(flags)
    if n >= ADVISORY_HIGH:
        return "high"
    if n >= ADVISORY_CAUTION:
        return "caution"
    return "none"
