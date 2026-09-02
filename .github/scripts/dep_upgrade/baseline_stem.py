#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Print the baseline_sbom stem a capture just recorded for an image:tag.

Reads the stem back out of the manifest rather than scraping capture logs.
container/context.yaml holds a single stem per framework and the licenses stage
appends -${TARGETARCH}, so per-arch stems that disagree are unrepresentable
downstream; this fails instead of shipping a half-right baseline.

Stdout: the bare stem, e.g. release@2366e4b4.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_MANIFEST = Path("container/compliance/base_sboms/manifest.json")


def stem_for(manifest: dict, image: str, tag: str, arches: list[str]) -> str:
    entries = manifest.get("entries", [])
    stems: dict[str, str] = {}
    for arch in arches:
        rows = [
            e
            for e in entries
            if e.get("from_image") == image
            and e.get("from_tag") == tag
            and e.get("platform") == f"linux/{arch}"
        ]
        if len(rows) != 1:
            raise SystemExit(
                f"expected exactly 1 {arch} row for {image}:{tag}, found {len(rows)}"
            )
        suffix = f"-{arch}.cdx.json"
        name = rows[0].get("baseline_sbom", "")
        if not name.endswith(suffix):
            raise SystemExit(f"{arch} baseline_sbom {name!r} does not end in {suffix}")
        stems[arch] = name[: -len(suffix)]
    if len(set(stems.values())) != 1:
        raise SystemExit(f"per-arch baseline stems disagree: {stems}")
    return next(iter(stems.values()))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True, help="from_image, e.g. nvcr.io/nvidia/x")
    p.add_argument("--tag", required=True, help="from_tag, e.g. 1.3.0rc25")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument(
        "--arches",
        default="amd64,arm64",
        help="comma-separated arches that must agree (default: %(default)s)",
    )
    args = p.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    arches = [a.strip() for a in args.arches.split(",") if a.strip()]
    print(stem_for(manifest, args.image, args.tag, arches))
    return 0


if __name__ == "__main__":
    sys.exit(main())
