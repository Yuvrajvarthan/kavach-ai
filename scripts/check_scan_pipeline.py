"""Run the full Scan pipeline outside the UI on all demo inputs.

    python scripts/check_scan_pipeline.py

Text messages -> run_scan; screenshot_*.png -> OCR -> run_scan; audio_*.wav -> Whisper -> run_scan.
Prints the extracted text and verdict headline for each, plus OCR/Whisper load and call times.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ocr import extract_text, get_ocr  # noqa: E402
from app.scan_core import run_scan  # noqa: E402
from app.speech import WHISPER_MODEL, get_whisper, transcribe  # noqa: E402

DEMO = ROOT / "demo_samples"


def show(source: str, text: str, extra: str = ""):
    s = run_scan(text)
    v, r = s["verdict"], s["result"]
    print(f"\n[{source}]{extra}")
    print(f"  text:    {text}")
    print(f"  verdict: {v['headline']}" + (f" | {v['detail']}" if v["detail"] else ""))
    print(f"  {v['footer']} · risk {r['risk']} · advisory {r['advisory']} · "
          f"flags: {', '.join(f['flag'] for f in r['red_flags']) or '-'}")


def main():
    print("=" * 30, "TEXT MESSAGES", "=" * 30)
    for m in json.loads((DEMO / "messages.json").read_text()):
        show(f"text {m['id']} (expected {m['expected']})", m["text"])

    print("\n" + "=" * 30, "SCREENSHOTS (RapidOCR)", "=" * 30)
    t0 = time.perf_counter()
    get_ocr()
    print(f"OCR engine load: {time.perf_counter() - t0:.2f}s")
    for p in sorted(DEMO.glob("screenshot_*.png")):
        t0 = time.perf_counter()
        text = extract_text(p.read_bytes())
        show(p.name, text, f"  OCR {time.perf_counter() - t0:.2f}s")

    print("\n" + "=" * 30, f"AUDIO (faster-whisper {WHISPER_MODEL}, CPU int8)", "=" * 30)
    t0 = time.perf_counter()
    get_whisper()
    print(f"Whisper first load: {time.perf_counter() - t0:.2f}s")
    for p in sorted(DEMO.glob("audio_*.wav")):
        t0 = time.perf_counter()
        tr = transcribe(p.read_bytes())
        show(p.name, tr["text"], f"  Whisper call {time.perf_counter() - t0:.2f}s · "
                                 f"spoken language detected: {tr['language']} ({tr['language_prob']:.2f})")


if __name__ == "__main__":
    main()
