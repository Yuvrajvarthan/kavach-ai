"""Vetted "What to do now" advice, as plain data.

Used by the Scan page and (Phase 5) the assistant's safety_guidance tool. The LLM must never
invent helplines: the only contacts here are the national cyber-crime helpline 1930,
https://cybercrime.gov.in, and the customer-care number printed on the user's own card.
"""
from __future__ import annotations

HELPLINE = "1930"
PORTAL = "https://cybercrime.gov.in"

CLASS_PLAIN = {
    "legit": "legitimate message",
    "government_impersonation": "government / police impersonation",
    "financial_fraud": "financial fraud (fake job, prize, loan or investment)",
    "bank_phishing": "bank phishing (fake KYC, card or account alert)",
    "romance_parcel_sextortion": "romance, parcel or sextortion scam",
}

REPORT_STEPS = [
    f"Report it: call the national cyber-crime helpline {HELPLINE} or file a complaint at {PORTAL}.",
    f"If you have already lost money, call {HELPLINE} as soon as possible and tell your bank.",
    "Keep screenshots of the message, the sender's number and any payment details as evidence.",
]

GENERAL_STEPS = [
    "Do not click links, call numbers or install apps mentioned in the message.",
    "Never share an OTP, PIN, CVV or password with anyone, including people claiming to be from a bank or the government.",
    "Do not pay any fee, fine or deposit that a message or caller demands.",
    "Verify independently: call your bank on the number printed on your card, or the organisation's official number.",
]

CLASS_STEPS = {
    "government_impersonation": [
        "Police, CBI or any other agency never arrest or question anyone over a video call. Disconnect the call.",
        "Do not pay any 'verification', 'bail' or 'clearance' amount, and do not share Aadhaar, PAN or bank details.",
        "Ignore instructions to stay on the call or to keep it secret, and tell a family member.",
    ],
    "financial_fraud": [
        "Do not pay a registration, processing or 'tax' fee to receive a job, prize, loan or refund.",
        "Guaranteed or very high returns are a classic sign of investment fraud. Do not transfer money to join a group.",
        "Do not share bank or UPI details with people you met through a message or social media.",
    ],
    "bank_phishing": [
        "Do not open the link or 'update KYC' from an SMS. Banks do not ask for your OTP, PIN or password.",
        "Call your bank on the number printed on your card to check whether there is any real problem.",
        "If you entered card or login details, ask your bank to block the card and change your passwords now.",
    ],
    "romance_parcel_sextortion": [
        "Do not pay customs, clearance or delivery charges for an unexpected gift or parcel.",
        "Do not send money to someone you have only met online, however urgent the story.",
        "If someone threatens to share private images: do not pay, stop replying, keep the evidence and report it.",
    ],
}

# When the model sees no scam pattern
LEGIT_TIPS = [
    "No scam pattern was detected, but stay alert: never share an OTP, PIN or password.",
    "If the message asks for money or personal details, verify with the sender through a number you already know.",
]


def get_guidance(label: str) -> dict:
    """Advice for a predicted class. For 'legit' returns gentle tips, otherwise class steps +
    general steps + how to report."""
    if label == "legit":
        return {"class": label, "title": "Stay alert", "steps": LEGIT_TIPS,
                "helpline": HELPLINE, "portal": PORTAL}
    return {
        "class": label,
        "title": f"What to do: {CLASS_PLAIN.get(label, label)}",
        "steps": CLASS_STEPS.get(label, []) + GENERAL_STEPS + REPORT_STEPS,
        "helpline": HELPLINE,
        "portal": PORTAL,
    }


def warning_signs_guidance() -> dict:
    """For messages the model calls legit but the rule layer flags (advisory high/caution)."""
    return {"class": None, "title": "What to do: treat it as a possible scam",
            "steps": GENERAL_STEPS + REPORT_STEPS, "helpline": HELPLINE, "portal": PORTAL}
