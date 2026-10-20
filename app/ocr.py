"""Screenshot -> text with RapidOCR (ONNX runtime, CPU). The engine loads on first use."""
from __future__ import annotations

import io
from functools import lru_cache

import numpy as np
from PIL import Image

# Lines that are phone status-bar / UI chrome rather than message text
_SKIP_PREFIXES = ("text message", "imessage", "sms", "delivered", "read ", "demo sample")


@lru_cache(maxsize=1)
def get_ocr():
    from rapidocr_onnxruntime import RapidOCR  # heavy import, keep lazy
    return RapidOCR()


def _is_chrome(line: str) -> bool:
    low = line.strip().lower()
    if not low:
        return True
    if low.startswith(_SKIP_PREFIXES):
        return True
    # status bar: clock "10:42", battery/network "5G 82%"
    return len(low) <= 10 and any(ch.isdigit() for ch in low) and (":" in low or "%" in low)


def extract_text(image: bytes | str | Image.Image) -> str:
    """OCR an image (bytes, path or PIL image) and return the message text, top to bottom."""
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))[:, :, ::-1]  # RGB -> BGR for OpenCV
    result, _ = get_ocr()(image)
    if not result:
        return ""
    lines = [r[1] for r in sorted(result, key=lambda r: (r[0][0][1], r[0][0][0]))]
    return " ".join(line for line in lines if not _is_chrome(line)).strip()
