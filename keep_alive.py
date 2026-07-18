#!/usr/bin/env python3
"""
keep_alive.py
Kohi Scheduler — daily keep-alive visit.

Streamlit Community Cloud puts an app to sleep after ~7 days without
viewer traffic, and only a real browser session counts as a viewer —
the daily weather-cache commit does not. This script opens the app in
headless Chromium (a real visit), clicks the "wake up" button if the
app is already asleep, and waits until the app actually renders.

Note: the wake-up button lives on the top-level page, but the running
app renders inside an iframe (URL path /~/+/), so both are searched
across all frames.

Run daily by .github/workflows/keep_alive.yml.
Requires: pip install playwright && playwright install chromium
"""

import sys
import time
from playwright.sync_api import sync_playwright

APP_URL = "https://kohi-scheduler.streamlit.app"
APP_READY_SELECTOR = '[data-testid="stAppViewContainer"]'
WAKE_TIMEOUT_S = 300  # cold wake-ups can take a few minutes


def find_in_any_frame(page, selector: str):
    for frame in page.frames:
        try:
            loc = frame.locator(selector)
            if loc.count() > 0:
                return loc.first
        except Exception:
            continue
    return None


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(APP_URL, wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_timeout(5_000)

        # Sleeping apps show a "Zzzz" page with a wake-up button.
        wake = find_in_any_frame(page, 'button:has-text("app back up")')
        if wake is not None:
            print("App was asleep — clicking the wake-up button…")
            wake.click()

        deadline = time.monotonic() + WAKE_TIMEOUT_S
        while time.monotonic() < deadline:
            if find_in_any_frame(page, APP_READY_SELECTOR) is not None:
                # Hold the session briefly so the visit registers.
                page.wait_for_timeout(10_000)
                print(f"OK — app is awake: {page.title()!r}")
                browser.close()
                return 0
            page.wait_for_timeout(3_000)

        print(f"ERROR: app did not render within {WAKE_TIMEOUT_S}s", file=sys.stderr)
        browser.close()
        return 1


if __name__ == "__main__":
    sys.exit(main())
