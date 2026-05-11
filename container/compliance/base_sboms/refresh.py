#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Refresh the base-image SBOM corpus.

Pulls each base image we build FROM, runs `syft scan -o cyclonedx-json`,
applies a slim filter, and writes one *.cdx.json per base into this
directory. Updates manifest.json to map image:tag@digest → filename.

The slim filter:
  - Drops `components[].properties` (syft's internal cataloger metadata,
    ~30% of the file size).
  - Drops `components[].hashes` (we trust the registry, not the SBOM,
    for tamper detection).
  - Drops `dependencies` (full transitive graph; we only need the flat
    component list for diffing against target images).
  - KEEPS `components[].evidence` — paths to where each package was
    found inside the image are critical for audit.

Hard cap: 4 MB per file. Exceeding it splits the SBOM per-ecosystem
(deb.cdx.json, pypi.cdx.json, rpm.cdx.json) under a directory named
after the base+digest prefix.

Runs in .github/workflows/base-sboms-refresh.yml weekly. The bot's PR
re-runs container/compliance/policy/validate.py against any new
licenses surfaced by base bumps; if validation fails (UNKNOWN, denied,
or new license category not in licenses.toml), the bot's CI blocks
the PR until a human updates licenses.toml.

TODO: implement (requires syft on the runner; bash-out + JSON post-process).
This is a stub at the right shape so the wiring around it (manifest,
verifier, CI workflow) can be built and tested. The actual refresh logic
is a separate focused commit when the cron workflow lands.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_CORPUS_DIR = Path(__file__).resolve().parent
_MANIFEST_PATH = _CORPUS_DIR / "manifest.json"
_SIZE_CAP_BYTES = 4 * 1024 * 1024  # 4 MB; 1 MB headroom under the 5 MB GitLab pain point


# Bases we currently build FROM. Refresh script pulls each, runs syft,
# slim-filters, and stores. New bases get added here.
DEFAULT_BASES: list[str] = [
    # CUDA family
    "nvcr.io/nvidia/cuda-dl-base:25.06-cuda12.9-devel-ubuntu24.04",
    "nvcr.io/nvidia/cuda-dl-base:25.06-cuda13.0-devel-ubuntu24.04",
    "nvcr.io/nvidia/cuda-dl-base:25.06-cuda13.1-devel-ubuntu24.04",
    # Framework bases
    "lmsysorg/sglang:v0.5.10.post1-cu129",
    "vllm/vllm-openai:v0.12.0",
    "nvcr.io/nvidia/tritonserver:25.07-trtllm-python-py3",
    # Distroless / lightweight
    "gcr.io/distroless/static:nonroot",
    "python:3.12-slim",
]


def slim_cyclonedx(doc: dict) -> dict:
    """Apply the slim filter described in the module docstring."""
    out = dict(doc)
    out.pop("dependencies", None)
    components = out.get("components", []) or []
    slimmed = []
    for c in components:
        c_out = {k: v for k, v in c.items() if k not in ("properties", "hashes")}
        slimmed.append(c_out)
    out["components"] = slimmed
    return out


def refresh_one(image_ref: str, output_dir: Path) -> tuple[str, Path]:
    """Pull `image_ref`, run syft, write slim CycloneDX, return (digest, file path).

    TODO: implement. Skeleton:
      1. `docker pull <image_ref>` (or skopeo inspect for digest).
      2. Resolve the digest of the manifest we just fetched.
      3. `syft scan <image_ref> -o cyclonedx-json` → JSON.
      4. Apply slim_cyclonedx().
      5. Write `<base-name>@<digest-prefix>.cdx.json` to output_dir.
      6. Verify size <= _SIZE_CAP_BYTES; if exceeded, split per-ecosystem.
      7. Return (digest, file path) for manifest.json update.
    """
    raise NotImplementedError(
        "Base-SBOM refresh not yet implemented. "
        "Run the syft pipeline manually for now."
    )


def update_manifest(entries: list[dict], path: Path) -> None:
    payload = {
        "schema_version": 1,
        "description": (
            "Maps base-image references (FROM <image>:<tag>) to slim CycloneDX "
            "SBOMs in this directory. Refreshed weekly by "
            "container/compliance/base_sboms/refresh.py."
        ),
        "format": (
            "CycloneDX 1.6 JSON, slim filter (drop properties/hashes/dependencies; "
            "keep evidence). Hard cap 4 MB per file in CI."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "entries": sorted(entries, key=lambda e: e["image"]),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=_CORPUS_DIR,
        help="Directory to write SBOM files into (default: this directory)",
    )
    parser.add_argument(
        "--bases",
        nargs="*",
        default=DEFAULT_BASES,
        help="Override the default base-image list (default: %(default)s)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    entries: list[dict] = []
    for image in args.bases:
        try:
            digest, sbom_path = refresh_one(image, args.corpus_dir)
            entries.append(
                {
                    "image": image,
                    "digest": digest,
                    "sbom": sbom_path.name,
                    "size_bytes": sbom_path.stat().st_size,
                }
            )
            logger.info("Refreshed %s -> %s", image, sbom_path.name)
        except NotImplementedError:
            raise
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to refresh %s: %s", image, exc)
            return 2

    update_manifest(entries, args.corpus_dir / "manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
