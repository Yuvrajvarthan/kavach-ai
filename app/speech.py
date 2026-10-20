"""Speech -> text with faster-whisper ("small" by default, CPU int8) and text -> speech with
edge-tts. Models load on first use.

Hinglish note: Whisper, left to auto-detect, writes Hindi/Hinglish speech in Devanagari (or,
for "base", Urdu) script, which the classifier (trained on Roman-script text) cannot read. Decoding
with language="en" plus a short, content-neutral Roman-script Hinglish prompt keeps the
output in Roman script. The prompt contains no scam words, so it cannot leak into the text.
"""
from __future__ import annotations

import asyncio
import io
import os
import re
from functools import lru_cache

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")  # small >> base on Hinglish
TTS_VOICE_EN = os.getenv("TTS_VOICE_EN", "en-IN-NeerjaNeural")
TTS_VOICE_HI = os.getenv("TTS_VOICE_HI", "hi-IN-SwaraNeural")
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_ROMAN_PROMPT = "Haan bhai, main theek hoon. Aap kaise ho? Kal milte hain, okay."


@lru_cache(maxsize=1)
def get_whisper():
    from faster_whisper import WhisperModel  # heavy import, keep lazy
    return WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")


def transcribe(audio: bytes | str) -> dict:
    """Transcribe audio (bytes or file path). Returns {"text", "language", "language_prob"}
    where language is Whisper's own detection of the spoken language."""
    src = io.BytesIO(audio) if isinstance(audio, bytes) else audio
    model = get_whisper()
    segments, info = model.transcribe(src, language="en", initial_prompt=_ROMAN_PROMPT,
                                      beam_size=5)
    text = " ".join(s.text.strip() for s in segments).strip()
    # detect the spoken language separately (for replying in the user's language later)
    if isinstance(src, io.BytesIO):
        src.seek(0)
    try:
        from faster_whisper.audio import decode_audio
        lang, prob, _ = model.detect_language(decode_audio(src))
    except Exception:  # noqa: BLE001 - language is informational only
        lang, prob = None, None
    return {"text": text, "language": lang, "language_prob": prob}


def tts_voice(text: str) -> str:
    """Hindi voice if the reply contains Devanagari, else Indian-English (also reads Hinglish)."""
    return TTS_VOICE_HI if _DEVANAGARI.search(text) else TTS_VOICE_EN


def _speakable(text: str, limit: int = 1500) -> str:
    t = re.sub(r"https?://", "", text)
    t = re.sub(r"[*_#`>|]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:limit]


def synthesize(text: str, timeout: float = 20.0) -> bytes | None:
    """Text -> MP3 bytes with edge-tts (needs internet). Returns None on any failure, so the
    caller can simply show the text."""
    text = _speakable(text)
    if not text:
        return None
    try:
        import edge_tts

        async def _run() -> bytes:
            buf = bytearray()
            async for chunk in edge_tts.Communicate(text, tts_voice(text)).stream():
                if chunk["type"] == "audio":
                    buf.extend(chunk["data"])
            return bytes(buf)

        audio = asyncio.run(asyncio.wait_for(_run(), timeout))
        return audio or None
    except Exception:  # noqa: BLE001 - offline / service error -> text only
        return None
