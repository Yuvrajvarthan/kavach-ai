"""Capture the app screenshots for the README / report with Playwright (dev-only).

    pip install -r requirements-dev.txt && playwright install chromium
    streamlit run app/Home.py            # in another terminal
    python scripts/take_screenshots.py [--url http://localhost:8501]

Saves 1440x900 PNGs to docs/screenshots/: home, scan_text, scan_screenshot,
model_performance, assistant_voice. The assistant shot calls the configured LLM.
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"
VIEWPORT = {"width": 1440, "height": 900}
BUSY = '[data-testid="stStatusWidget"], [data-testid="stSpinner"]'


def wait_idle(page: Page, timeout_s: float = 60, settle_s: float = 1.5):
    """Wait until no Streamlit status widget / spinner has been visible for `settle_s`.
    (With toolbarMode=minimal the status widget may be hidden, so long steps also wait for
    their own result element.)"""
    page.wait_for_selector('[data-testid="stApp"]', timeout=timeout_s * 1000)
    end = time.monotonic() + timeout_s
    quiet_since = None
    while time.monotonic() < end:
        if page.locator(BUSY).count() == 0:
            quiet_since = quiet_since or time.monotonic()
            if time.monotonic() - quiet_since >= settle_s:
                return
        else:
            quiet_since = None
        page.wait_for_timeout(250)
    raise TimeoutError("app did not become idle")


def pick(page: Page, label: str, option: str):
    """Choose an option in the st.selectbox whose label contains `label`."""
    box = page.locator('[data-testid="stSelectbox"]').filter(has_text=label)
    box.click()
    page.keyboard.type(option)
    page.wait_for_timeout(300)
    page.keyboard.press("Enter")


_SCROLL_JS = """(e, off) => {
    e.scrollIntoView({block: "start"});
    let p = e.parentElement;
    while (p && !(p.scrollHeight > p.clientHeight && /auto|scroll/.test(getComputedStyle(p).overflowY)))
        p = p.parentElement;
    (p || document.scrollingElement).scrollBy(0, -off);
}"""


def scroll_to(page: Page, text: str | None = None, offset: int = 90, locator=None):
    """Scroll Streamlit's inner scroll container so the element sits `offset` px from the top."""
    target = locator if locator is not None else page.get_by_text(text, exact=False).first
    target.evaluate(_SCROLL_JS, offset)
    page.wait_for_timeout(600)


def shot(page: Page, name: str):
    path = OUT / name
    page.screenshot(path=str(path))
    print(f"saved {path.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    url = ap.parse_args().url.rstrip("/")
    OUT.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT)

        page.goto(url)
        wait_idle(page)
        shot(page, "home.png")

        page.goto(f"{url}/Scan_Message")
        wait_idle(page)
        pick(page, "Try a sample message", "hing_01")
        wait_idle(page)
        scroll_to(page, "Be careful:", offset=40)
        shot(page, "scan_text.png")

        page.goto(f"{url}/Scan_Message")
        wait_idle(page)
        page.get_by_role("tab", name=re.compile("Screenshot")).click()
        page.locator('[data-testid="stFileUploader"]').filter(has_text="Upload a screenshot") \
            .locator('input[type="file"]').set_input_files(str(ROOT / "demo_samples/screenshot_gov_01.png"))
        wait_idle(page, timeout_s=90)
        page.get_by_role("button", name="Scan extracted text").click()
        wait_idle(page)
        scroll_to(page, "Text found in the screenshot", offset=60)  # image + OCR text + verdict
        shot(page, "scan_screenshot.png")

        page.goto(f"{url}/Model_Performance")
        wait_idle(page)
        shot(page, "model_performance.png")

        page.goto(f"{url}/Kavach_Assistant")
        wait_idle(page)
        page.get_by_text("Ask by voice").click()
        page.wait_for_timeout(800)  # let the expander open before using its selectbox
        pick(page, "use a demo voice question", "audio_query_hinglish")
        page.locator('[data-testid="stChatMessage"]').first.wait_for(timeout=120_000)  # transcript
        # Whisper + LLM + TTS: wait for the assistant's reply bubble itself
        page.locator('[data-testid="stChatMessage"]').nth(1).wait_for(timeout=240_000)
        wait_idle(page, timeout_s=60, settle_s=2.5)
        scroll_to(page, locator=page.locator('[data-testid="stChatMessage"]').first, offset=20)
        shot(page, "assistant_voice.png")

        browser.close()


if __name__ == "__main__":
    main()
