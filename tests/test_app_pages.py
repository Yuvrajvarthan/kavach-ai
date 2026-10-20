"""Streamlit AppTest: every page loads without exceptions, and the Scan page's text flow
produces the right verdict card for a scam, a rule-flagged message and a legit one.

    python -m pytest tests/test_app_pages.py -q
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["app/Home.py", "app/pages/1_Scan_Message.py", "app/pages/2_Model_Performance.py",
         "app/pages/3_Kavach_Assistant.py"]
SCAN = str(ROOT / "app/pages/1_Scan_Message.py")
ASSISTANT = str(ROOT / "app/pages/3_Kavach_Assistant.py")
DEMO = {m["id"]: m["text"] for m in json.loads((ROOT / "demo_samples/messages.json").read_text())}


@pytest.mark.parametrize("page", PAGES)
def test_page_loads(page):
    at = AppTest.from_file(str(ROOT / page), default_timeout=60).run()
    assert not at.exception, at.exception


def _scan_text(text: str) -> str:
    at = AppTest.from_file(SCAN, default_timeout=60).run()
    at.text_area(key="text_input").set_value(text).run()
    at.button(key="scan_text_btn").click().run()
    assert not at.exception, at.exception
    return "\n".join(m.value for m in at.markdown)


def test_uncertain_scam_notes():
    page = _scan_text("Hi, can you call me when you are free? Need to discuss the project today.")
    assert "Likely scam: romance, parcel or sextortion scam" in page
    assert "The model isn't sure which scam type this is (top match 38%)." in page
    assert "If you know the sender, confirm with them on a number you already have." in page


def test_assistant_offline_chat(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:9/v1")  # connection refused
    at = AppTest.from_file(ASSISTANT, default_timeout=60).run()
    at.toggle[0].set_value(False).run()  # no TTS network call in tests
    at.chat_input[0].set_value(DEMO["bank_01"]).run()
    assert not at.exception, at.exception
    page = "\n".join(m.value for m in at.markdown) + "\n".join(c.value for c in at.caption)
    assert "Likely scam: bank phishing" in page
    assert "Offline mode" in page


@pytest.mark.parametrize("sample_id, headline", [
    ("gov_01", "Likely scam: government / police impersonation"),
    ("hing_01", "Be careful: 4 warning signs found"),
    ("legit_04", "No scam pattern detected"),
])
def test_scan_verdicts(sample_id, headline):
    page = _scan_text(DEMO[sample_id])
    assert headline in page
    assert "Model: " in page and "scam probability" in page
    assert ">Safe<" not in page


def test_sample_picker_fills_and_scans():
    at = AppTest.from_file(SCAN, default_timeout=60).run()
    option = next(o for o in at.selectbox(key="sample_choice").options if o.startswith("bank_01"))
    at.selectbox(key="sample_choice").set_value(option).run()
    assert not at.exception, at.exception
    assert at.text_area(key="text_input").value == DEMO["bank_01"]
    assert "Likely scam: bank phishing" in "\n".join(m.value for m in at.markdown)
