#!/usr/bin/env python3
#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""Example: HTML+CSS -> PNG comparison cards (Dynamo Dark, compact title).

Renders the sibling ``fig-cards.html`` to PNG with Playwright — the fifth
pathway from the skill, for typography-heavy card layouts where CSS is the
most expressive layout engine. The HTML is the source; this script is just
the deterministic Playwright renderer (`.figure` element screenshotted at
2x).

Demonstrates: the **compact / chart title** treatment in CSS (`h1`: Arial,
weight 700, uppercase, 0.08em tracking), token surfaces/borders, square
corners, and a single green accent card. Card content is representative,
hard-coded in the HTML (no external data).

Prerequisite: ``pip install playwright && playwright install chromium``.

Usage:
    python3 gen_fig_cards.py                 # -> images/fig-cards.png
"""

from __future__ import annotations

from pathlib import Path

HTML_SRC = Path(__file__).parent / "fig-cards.html"


def main() -> None:
    png_path = Path(__file__).parent / "images" / "fig-cards.png"
    png_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "SKIP fig-cards.png: playwright not installed "
            "(`pip install playwright && playwright install chromium`)"
        )
        return

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(
                viewport={"width": 1200, "height": 800},
                device_scale_factor=2,
            )
            page.goto(f"file://{HTML_SRC.resolve()}")
            page.locator(".figure").screenshot(
                path=str(png_path), omit_background=False
            )
            browser.close()
    except Exception as exc:  # noqa: BLE001 - render env may lack chromium
        print(
            f"SKIP fig-cards.png: Playwright render failed ({exc}). "
            "Run `playwright install chromium`."
        )
        return

    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
