"""Speech -> text with faster-whisper ("small" by default, CPU int8) and text -> speech with
edge-tts, falling back to the built-in macOS voices (`say`) when there is no internet.
Models load on first use.

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
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")  # small >> base on Hinglish
TTS_VOICE_EN = os.getenv("TTS_VOICE_EN", "en-IN-NeerjaNeural")
TTS_VOICE_HI = os.getenv("TTS_VOICE_HI", "hi-IN-SwaraNeural")
SAY_VOICE_EN = os.getenv("SAY_VOICE_EN", "Rishi")  # offline macOS voices (English India / Hindi)
SAY_VOICE_HI = os.getenv("SAY_VOICE_HI", "Lekha")
# KAVACH_OFFLINE_VOICE=1 skips edge-tts (internet) and always uses the macOS voices.
_edge_tts_ok = os.getenv("KAVACH_OFFLINE_VOICE", "0") != "1"
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_ROMAN_PROMPT = "Haan bhai, main theek hoon. Aap kaise ho? Kal milte hain, okay."


@lru_cache(maxsize=1)
def get_whisper():
    from faster_whisper import WhisperModel  # heavy import, keep lazy
    try:  # load from the local cache without touching the network (works with Wi-Fi off)
        return WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", local_files_only=True)
    except Exception:  # noqa: BLE001 - first run: not cached yet, download once
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


def _say_offline(text: str, timeout: float = 60.0) -> bytes | None:
    """Text -> WAV bytes with the macOS `say` command (no internet needed). None if unavailable."""
    if not shutil.which("say"):
        return None
    voice = SAY_VOICE_HI if _DEVANAGARI.search(text) else SAY_VOICE_EN
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "reply.wav"
        try:
            subprocess.run(["say", "-v", voice, "-o", str(out), "--file-format=WAVE",
                            "--data-format=LEI16@22050", text],
                           check=True, timeout=timeout, capture_output=True)
            return out.read_bytes() or None
        except Exception:  # noqa: BLE001 - voice missing / timeout -> text only
            return None


def synthesize(text: str, timeout: float = 8.0) -> bytes | None:
    """Text -> audio bytes: edge-tts MP3 when online, else macOS `say` WAV (offline).
    Returns None if both fail, so the caller can simply show the text."""
    global _edge_tts_ok
    text = _speakable(text)
    if not text:
        return None
    if not _edge_tts_ok:
        return _say_offline(text)
    try:
        import edge_tts

        async def _run() -> bytes:
            buf = bytearray()
            async for chunk in edge_tts.Communicate(text, tts_voice(text)).stream():
                if chunk["type"] == "audio":
                    buf.extend(chunk["data"])
            return bytes(buf)

        audio = asyncio.run(asyncio.wait_for(_run(), timeout))
        if audio:
            return audio
    except Exception:  # noqa: BLE001 - offline / service error -> offline voice from now on
        _edge_tts_ok = False
    return _say_offline(text)


def audio_format(audio: bytes) -> str:
    """MIME type for st.audio: WAV from the offline voice, MP3 from edge-tts."""
    return "audio/wav" if audio[:4] == b"RIFF" else "audio/mp3"
