import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from app.common import CLASS_COLORS, demo_files, get_demo_messages, get_model, page_setup  # noqa: E402
from app.scan_core import highlight_html, run_scan  # noqa: E402
from src.guidance import CLASS_PLAIN, HELPLINE, PORTAL  # noqa: E402
from src.preprocess import CLASSES  # noqa: E402

page_setup("Scan a Message", "🔍")
st.title("🔍 Scan a Message")
st.caption("Paste the text, upload a screenshot, or speak / upload a voice note. "
           "Nothing you scan is stored.")


@st.cache_data(show_spinner=False, max_entries=200)
def cached_scan(text: str) -> dict:
    return run_scan(text, model=get_model())


@st.cache_data(show_spinner=False, max_entries=50)
def cached_ocr(image_bytes: bytes) -> str:
    from app.ocr import extract_text
    return extract_text(image_bytes)


@st.cache_data(show_spinner=False, max_entries=50)
def cached_transcribe(audio_bytes: bytes) -> dict:
    from app.speech import transcribe
    return transcribe(audio_bytes)


def _digest(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


# ------------------------------------------------------------------ sample picker
samples = get_demo_messages()
sample_labels = {f"{m['id']} · {m['text'][:70]}…": m for m in samples}


def _load_sample():
    choice = st.session_state.get("sample_choice")
    if choice in sample_labels:
        st.session_state["text_input"] = sample_labels[choice]["text"]
        st.session_state["scan_query"] = sample_labels[choice]["text"]


st.selectbox("🧪 Try a sample message", ["—"] + list(sample_labels), key="sample_choice",
             on_change=_load_sample,
             help="Fictional test messages from demo_samples/messages.json")

tab_text, tab_shot, tab_voice = st.tabs(["✍️ Text", "🖼️ Screenshot", "🎙️ Voice / audio"])

# ------------------------------------------------------------------ text
with tab_text:
    st.text_area("Message text", key="text_input", height=140,
                 placeholder="Paste the suspicious SMS or WhatsApp message here…")
    if st.button("Scan message", type="primary", key="scan_text_btn"):
        st.session_state["scan_query"] = st.session_state.get("text_input", "")

# ------------------------------------------------------------------ screenshot
with tab_shot:
    shots = demo_files("screenshot_*.png")
    c1, c2 = st.columns(2)
    upload = c1.file_uploader("Upload a screenshot", type=["png", "jpg", "jpeg", "webp"])
    demo_shot = c2.selectbox("…or use a demo screenshot", ["—"] + [p.name for p in shots])
    image_bytes = None
    if upload is not None:
        image_bytes = upload.getvalue()
    elif demo_shot != "—":
        image_bytes = (shots[0].parent / demo_shot).read_bytes()
    if image_bytes:
        digest = _digest(image_bytes)
        if st.session_state.get("ocr_digest") != digest:
            with st.spinner("Reading text from the image… (the first time also loads the OCR engine)"):
                st.session_state["ocr_text"] = cached_ocr(image_bytes)
            st.session_state["ocr_digest"] = digest
        cimg, ctxt = st.columns([1, 2.4])
        cimg.image(image_bytes, width=260)
        with ctxt:
            st.text_area("Text found in the screenshot (edit if needed)", key="ocr_text", height=180)
            if not st.session_state.get("ocr_text"):
                st.warning("No text was found in this image.")
            if st.button("Scan extracted text", type="primary", key="scan_ocr_btn"):
                st.session_state["scan_query"] = st.session_state.get("ocr_text", "")

# ------------------------------------------------------------------ voice
with tab_voice:
    audios = demo_files("audio_gov_01.wav")  # audio_query_* clips are questions -> Assistant page
    c1, c2 = st.columns(2)
    recorded = c1.audio_input("Record a voice note (read the message aloud or ask about it)")
    uploaded_audio = c2.file_uploader("…or upload an audio file",
                                      type=["wav", "mp3", "m4a", "ogg", "webm", "flac"])
    demo_audio = c2.selectbox("…or use a demo clip (a scam message read aloud)",
                              ["—"] + [p.name for p in audios])
    audio_bytes = None
    if recorded is not None:
        audio_bytes = recorded.getvalue()
    elif uploaded_audio is not None:
        audio_bytes = uploaded_audio.getvalue()
    elif demo_audio != "—":
        audio_bytes = (audios[0].parent / demo_audio).read_bytes()
    if audio_bytes:
        st.audio(audio_bytes)
        digest = _digest(audio_bytes)
        if st.session_state.get("voice_digest") != digest:
            with st.spinner("Transcribing… (the first time also loads the Whisper speech model)"):
                st.session_state["voice_text"] = cached_transcribe(audio_bytes)["text"]
            st.session_state["voice_digest"] = digest
        st.text_area("Transcript (edit if needed)", key="voice_text", height=120)
        st.caption("Speech is written in Roman script so the model can read Hinglish. "
                   "Check the transcript before scanning.")
        if st.button("Scan transcript", type="primary", key="scan_voice_btn"):
            st.session_state["scan_query"] = st.session_state.get("voice_text", "")

# ------------------------------------------------------------------ result
query = (st.session_state.get("scan_query") or "").strip()
if not query:
    st.info("Choose a sample above, or enter a message and press **Scan**.")
    st.stop()

with st.spinner("Scanning…"):
    scan = cached_scan(query)
res, v = scan["result"], scan["verdict"]

st.divider()
detail = f'<div style="margin-top:4px;font-size:1rem">{v["detail"]}</div>' if v["detail"] else ""
st.markdown(
    f'<div style="border-left:8px solid {v["accent"]};background:{v["bg"]};color:{v["fg"]};'
    f'padding:16px 20px;border-radius:8px">'
    f'<div style="font-size:1.45rem;font-weight:700">{v["headline"]}</div>{detail}'
    f'<div style="margin-top:10px;font-size:0.8rem;opacity:0.8">{v["footer"]}</div></div>',
    unsafe_allow_html=True)
for note in v["notes"]:
    st.markdown(f"ℹ️ {note}")

left, right = st.columns([1, 1.3], gap="large")
with left:
    st.markdown("#### Model probabilities")
    probs = res["probabilities"]
    order = CLASSES[::-1]
    fig = go.Figure(go.Bar(
        x=[probs[c] for c in order], y=[CLASS_PLAIN[c].split(" (")[0] for c in order],
        orientation="h", marker_color=[CLASS_COLORS[c] for c in order],
        text=[f"{probs[c]:.0%}" for c in order], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.update_layout(height=260, margin=dict(l=10, r=50, t=10, b=30),
                      xaxis=dict(range=[0, 1.15], tickformat=".0%"))
    st.plotly_chart(fig, width="stretch")
with right:
    st.markdown("#### Why: words that drove the prediction")
    st.markdown(f'<div style="line-height:1.9;font-size:1.02rem">'
                f'{highlight_html(scan["text"], scan["highlight_words"])}</div>',
                unsafe_allow_html=True)
    toward = CLASS_PLAIN.get(scan["highlight_class"], scan["highlight_class"])
    st.caption(f"🟥 pushes toward *{toward}* · 🟩 pushes away from it (toward legit). "
               "Word weights from LIME; hover a word to see its weight.")

st.markdown("#### 🚩 Red-flag checklist")
c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown("**Basic signals**")
    for s in scan["signals"]:
        mark = "🔴" if s["evidence"] else "⚪"
        ev = f" — `{'`, `'.join(s['evidence'][:4])}`" if s["evidence"] else ""
        st.markdown(f"{mark} {s['label']}{ev}")
with c2:
    st.markdown(f"**Scam tactics (rule layer)** · advisory: `{res['advisory']}`")
    if res["red_flags"]:
        for f in res["red_flags"]:
            st.markdown(f"🔴 {f['label']} — `{'`, `'.join(f['evidence'][:4])}`")
    else:
        st.markdown("⚪ None of the 9 scam tactics found")
    st.caption("Rules are separate from the ML model and never change its prediction.")

g = scan["guidance"]
st.markdown(f"#### 🛡️ What to do now")
box = st.error if res["label"] != "legit" else (st.warning if res["advisory"] != "none" else st.success)
box("\n".join(f"{i}. {s}" for i, s in enumerate(g["steps"], 1)))
st.caption(f"National cyber-crime helpline: **{HELPLINE}** · [cybercrime.gov.in]({PORTAL})")
