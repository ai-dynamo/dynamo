#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Regenerate components/publisher-logos.generated.ts.

Fetches each organisation's own site icon, downscales it to 48x48 and inlines
it as a data URI, keyed by the name used in PUBLICATIONS.partner.

Keyed by organisation, not by the host of the article: several of these pieces
are published somewhere other than the author's own site (Doubleword on
TelecomTV, Deloitte on LinkedIn), and keying on the host puts the wrong
company's mark on the card.

Data URIs rather than files under assets/ because Fern rewrites asset paths
only in MDX and docs.yml -- a repo path written in a .tsx reaches the browser
verbatim and 404s once published. See check_asset_paths.py.

Not every site serves an icon to a script: AstraZeneca answers 403 to every
path tried. Anything missing here falls back to the publisher's initials in
the component, so partial coverage renders correctly.

Adding a publisher: add it to SOURCES (or ICON_OVERRIDES if the site's
<link rel="icon"> is unusable), then run this and commit the result.

Usage: python3 generate_publisher_logos.py [--check]
  --check  exit 1 if the committed file is out of date

Requires: pillow. Network access.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import subprocess
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # docs/fern
OUT = ROOT / "components" / "publisher-logos.generated.ts"
SIZE = 48
UA = "Mozilla/5.0"

# Organisation -> the site whose icon represents it.
SOURCES: dict[str, str] = {
    "AWS": "aws.amazon.com",
    "Alibaba Cloud / ACK": "www.alibabacloud.com",
    "Alibaba Cloud community": "www.alibabacloud.com",
    "Amazon Ads": "advertising.amazon.com",
    "AstraZeneca": "www.astrazeneca.com",
    "Azure Global Black Belt": "azure.microsoft.com",
    "Baseten": "www.baseten.co",
    "ClearML": "clear.ml",
    "Cognition": "cognition.com",
    "CoreWeave": "www.coreweave.com",
    "Crusoe": "www.crusoe.ai",
    "Deloitte": "www2.deloitte.com",
    "DigitalOcean": "www.digitalocean.com",
    "DigitalOcean / Workato": "www.digitalocean.com",
    "Doubleword": "doubleword.ai",
    "Everpure / Pure Storage": "www.purestorage.com",
    "GMI Cloud": "www.gmicloud.ai",
    "Gcore": "gcore.com",
    "Google Cloud": "cloud.google.com",
    "H Company": "hcompany.ai",
    "Hao AI Lab / UCSD": "haoailab.com",
    "Intel": "www.intel.com",
    "LMCache": "lmcache.ai",
    "LMSYS / SGLang": "www.lmsys.org",
    "Microsoft Azure": "azure.microsoft.com",
    "Microsoft Azure / AKS": "azure.microsoft.com",
    "OpenNebula": "opennebula.io",
    "Photoroom": "www.photoroom.com",
    "Prime Intellect": "www.primeintellect.ai",
    "Rafay": "rafay.co",
    "SemiAnalysis / InferenceX": "semianalysis.com",
    "SkyPilot": "skypilot.co",
    "Spheron": "www.spheron.network",
    "Together AI": "www.together.ai",
    "VAST Data": "www.vastdata.com",
    "Vultr": "www.vultr.com",
    "WEKA": "www.weka.io",
    "dstack": "dstack.ai",
    "vCluster": "www.vcluster.com",
}

# Sites whose <link rel="icon"> is missing or unusable; fetch these directly.
ICON_OVERRIDES: dict[str, str] = {
    "SkyPilot": "https://avatars.githubusercontent.com/u/109387420?v=4"
}


def fetch(url: str) -> bytes:
    try:
        return subprocess.run(
            ["curl", "-sSL", "--max-time", "20", "-A", UA, url],
            capture_output=True,
            timeout=30,
        ).stdout
    except Exception:
        return b""


def icon_candidates(domain: str) -> list[str]:
    """Best icon URLs for a domain, largest declared size first."""
    page = f"https://{domain}/"
    html = fetch(page)[:400000].decode("utf-8", "ignore")
    found: list[tuple[int, str]] = []
    for tag in re.findall(r"<link[^>]+>", html, re.I):
        if not re.search(r'rel=["\']?[^"\'>]*icon', tag, re.I):
            continue
        href = re.search(r'href=["\']([^"\']+)["\']', tag)
        if not href:
            continue
        sized = re.search(r'sizes=["\'](\d+)', tag)
        rank = int(sized.group(1)) if sized else (180 if "apple" in tag.lower() else 0)
        found.append((rank, urllib.parse.urljoin(page, href.group(1))))
    found.sort(reverse=True)
    return [u for _, u in found] + [f"https://{domain}/favicon.ico"]


def encode(raw: bytes) -> str | None:
    """Downscale to SIZE and return a PNG data URI, or None if unreadable."""
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGBA")
    except Exception:
        return None
    if min(image.size) < 16:
        return None
    image.thumbnail((SIZE, SIZE), Image.LANCZOS)
    canvas = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    canvas.paste(image, ((SIZE - image.width) // 2, (SIZE - image.height) // 2), image)
    buffer = io.BytesIO()
    canvas.save(buffer, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def collect() -> tuple[dict[str, str], list[str]]:
    logos: dict[str, str] = {}
    missing: list[str] = []
    for name, domain in sorted(SOURCES.items()):
        if name in ICON_OVERRIDES:
            urls = [ICON_OVERRIDES[name]]
        else:
            # Keep the /favicon.ico fallback (last) even when the page declares
            # several <link rel="icon"> entries -- capping the list without it
            # loses sites whose declared icons all fail.
            declared = icon_candidates(domain)
            urls = list(dict.fromkeys(declared[:3] + declared[-1:]))
        for url in urls:
            raw = fetch(url)
            if len(raw) < 200:
                continue
            data = encode(raw)
            if data:
                logos[name] = data
                break
        else:
            missing.append(name)
    return logos, missing


BANNER = """/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * AUTO-GENERATED by docs/fern/scripts/generate_publisher_logos.py -- do not
 * edit by hand. Run that script to add a publisher or refresh the set.
 *
 * Publisher logos keyed by the organisation credited in PUBLICATIONS.partner,
 * each taken from that organisation's own site, downscaled to 48x48 and
 * inlined as a data URI. A partner with no entry here falls back to its
 * initials in the component, so partial coverage renders correctly.
 */

export const PUBLISHER_LOGOS: Record<string, string> = {
"""


def render(logos: dict[str, str]) -> str:
    lines = [
        f"  {json.dumps(k)}: {json.dumps(v)},"
        for k in sorted(logos)
        for v in [logos[k]]
    ]
    return BANNER + "\n".join(lines) + "\n};\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    logos, missing = collect()
    rendered = render(logos)

    if args.check:
        if not OUT.exists() or OUT.read_text() != rendered:
            print(
                f"{OUT.relative_to(ROOT.parents[1])} is out of date; rerun without --check"
            )
            return 1
        print(f"publisher logos up to date: {len(logos)} of {len(SOURCES)}")
        return 0

    OUT.write_text(rendered)
    print(
        f"wrote {OUT.relative_to(ROOT.parents[1])}: {len(logos)} of {len(SOURCES)} publishers"
    )
    if missing:
        print("  no icon served (falls back to initials): " + ", ".join(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
