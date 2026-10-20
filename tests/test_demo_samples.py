"""Run the trained classifier + red-flag rules on demo_samples/messages.json; print predicted vs expected.

    python -m pytest tests/test_demo_samples.py -s      (or)      python tests/test_demo_samples.py

`expected` is either a class name, or e.g. "scam (financial_fraud)" for the Hinglish samples,
where only scam-vs-legit is required and the classes in brackets are acceptable.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predict import analyse  # noqa: E402
from src.preprocess import CLASSES  # noqa: E402

SAMPLES = ROOT / "demo_samples" / "messages.json"


def parse_expected(expected: str) -> tuple[bool, set[str]]:
    """-> (is_scam, acceptable class names)."""
    if expected in CLASSES:
        return expected != "legit", {expected}
    return True, set(re.findall(r"[a-z_]+", expected)) & set(CLASSES)


def run_samples() -> list[dict]:
    rows = []
    for m in json.loads(SAMPLES.read_text()):
        res = analyse(m["text"])
        is_scam, ok_classes = parse_expected(m["expected"])
        rows.append({
            "id": m["id"], "expected": m["expected"], "predicted": res["label"],
            "scam_prob": res["scam_prob"], "risk": res["risk"],
            "advisory": res["advisory"], "flags": [f["flag"] for f in res["red_flags"]],
            "class_ok": res["label"] in ok_classes,
            "binary_ok": (res["label"] != "legit") == is_scam,
        })
    return rows


def print_table(rows: list[dict]) -> None:
    print(f"\n{'id':<11}{'expected':<44}{'ML label':<27}{'scam_p':>7} {'risk':<7}{'class':>6}"
          f"{'binary':>7}  {'advisory':<9}flags")
    for r in rows:
        print(f"{r['id']:<11}{r['expected'][:43]:<44}{r['predicted']:<27}{r['scam_prob']:>7.3f} "
              f"{r['risk']:<7}{'✓' if r['class_ok'] else '✗':>6}{'✓' if r['binary_ok'] else '✗':>7}  "
              f"{r['advisory']:<9}{', '.join(r['flags']) or '-'}")
    n = len(rows)
    print(f"\nclass correct: {sum(r['class_ok'] for r in rows)}/{n}   "
          f"scam-vs-legit correct: {sum(r['binary_ok'] for r in rows)}/{n}")


def test_demo_samples():
    rows = run_samples()
    print_table(rows)
    for r in rows:
        assert r["predicted"] in CLASSES
        assert 0.0 <= r["scam_prob"] <= 1.0
        assert r["risk"] in {"High", "Medium", "Low"}
        assert r["advisory"] in {"high", "caution", "none"}


if __name__ == "__main__":
    print_table(run_samples())
