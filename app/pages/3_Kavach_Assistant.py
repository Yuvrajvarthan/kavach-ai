import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from app.common import demo_files, page_setup  # noqa: E402
from app.llm import llm_config, run_assistant  # noqa: E402
from app.scan_core import build_verdict  # noqa: E402

page_setup("Kavach Assistant", "🛡️")
cfg = llm_config()
st.title("🛡️ Kavach Assistant")
st.caption(f"Ask in English, Hindi or Hinglish, by text or voice. The trained ML model classifies "
           f"every message; the AI ({cfg['provider']} · {cfg['model'] or 'not configured'}) only "
           f"explains the result.")

if "chat" not in st.session_state:
    st.session_state.chat = []  # [{"role", "content", "voice", "result", "audio"}]

c1, c2 = st.columns([3, 1])
speak = c2.toggle("🔊 Speak replies", value=True)
if c2.button("Clear chat"):
    st.session_state.chat = []
    st.session_state.pending_turn = None
    st.session_state.pop("voice_digest_asst", None)


def _verdict_card(out: dict):
    v = build_verdict(out)
    st.markdown(
        f'<div style="border-left:6px solid {v["accent"]};background:{v["bg"]};color:{v["fg"]};'
        f'padding:10px 14px;border-radius:6px;margin-bottom:8px">'
        f'<div style="font-weight:700">{v["headline"]}</div>'
        + (f'<div style="font-size:0.9rem">{v["detail"]}</div>' if v["detail"] else "")
        + f'<div style="font-size:0.75rem;opacity:0.8;margin-top:4px">{v["footer"]} · '
        f'classified by the trained model</div></div>', unsafe_allow_html=True)


def _render(msg: dict):
    with st.chat_message(msg["role"], avatar="🛡️" if msg["role"] == "assistant" else None):
        if msg["role"] == "user":
            st.markdown(("🎙️ " if msg.get("voice") else "") + msg["content"])
            if msg.get("voice"):
                st.caption("Transcribed by Whisper")
            return
        r = msg["result"]
        for c in r.tool_calls:
            if c["name"] == "classify_message":
                _verdict_card(c["output"])
        st.markdown(msg["content"])
        if r.mode == "offline":
            st.caption("⚠️ Offline mode: the AI assistant is unavailable, so this answer was built "
                       "directly from the model output and vetted guidance.")
        elif r.mode == "guarded":
            st.caption("🛡️ The AI's reply was replaced by a checked answer built from the model "
                       "output and vetted guidance.")
        if msg.get("audio"):
            st.audio(msg["audio"], format="audio/mp3")
        with st.expander(f"🔧 Tools called ({len(r.tool_calls)}): the model, not the AI, classified"):
            st.caption(f"Mode: {r.mode} · language: {r.language} · provider: {r.provider}")
            if r.error:
                st.caption(f"Fallback reason: {r.error}")
            for c in r.tool_calls:
                tag = " · run by Kavach before the AI" if c.get("forced") else " · called by the AI"
                st.markdown(f"**`{c['name']}`**{tag}")
                st.json({"args": c["args"], "output": c["output"]}, expanded=False)
            if r.rejected_reply:
                st.markdown("**Rejected AI reply** (failed a safety check):")
                st.text(r.rejected_reply)


for m in st.session_state.chat:
    _render(m)

# ------------------------------------------------------------------ voice input
with c1.expander("🎙️ Ask by voice", expanded=False):
    v1, v2 = st.columns(2)
    recorded = v1.audio_input("Record your question")
    demo_clips = demo_files("audio_query_*.wav")
    demo_choice = v2.selectbox("…or use a demo voice question", ["—"] + [p.name for p in demo_clips],
                               key="demo_voice")

audio_bytes = recorded.getvalue() if recorded is not None else None
if audio_bytes is None and demo_choice != "—":
    audio_bytes = (demo_clips[0].parent / demo_choice).read_bytes()

pending, from_voice = None, False
if audio_bytes:
    digest = hashlib.sha1(audio_bytes).hexdigest()
    if st.session_state.get("voice_digest_asst") != digest:
        st.session_state["voice_digest_asst"] = digest
        with st.spinner("Transcribing your voice note… (the first time also loads Whisper)"):
            from app.speech import transcribe
            pending, from_voice = transcribe(audio_bytes)["text"].strip(), True
        if not pending:
            st.warning("Couldn't hear any words in that recording. Please try again.")

typed = st.chat_input("Paste a suspicious message or ask a question…")
if typed:
    pending, from_voice = typed.strip(), False

if pending:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat]
    user_msg = {"role": "user", "content": pending, "voice": from_voice}
    st.session_state.chat.append(user_msg)
    # Saved before the slow work: if a widget event interrupts this run, the next run answers it.
    st.session_state.pending_turn = {"text": pending, "history": history}
    _render(user_msg)

if st.session_state.get("pending_turn"):
    turn = st.session_state.pending_turn
    with st.spinner("Checking with the Kavach model and preparing an answer…"):
        result = run_assistant(turn["text"], turn["history"])
        audio = None
        if speak:
            from app.speech import synthesize
            audio = synthesize(result.reply)  # None if offline -> text only
        st.session_state.chat.append({"role": "assistant", "content": result.reply,
                                      "result": result, "audio": audio})
        st.session_state.pending_turn = None
    st.rerun()

if not st.session_state.chat:
    st.info("Try: paste a message you received, ask *“How many bank phishing messages are in the "
            "dataset?”*, or *“Mujhe CBI officer ne video call pe arrest karne ki dhamki di, kya "
            "karun?”*")
