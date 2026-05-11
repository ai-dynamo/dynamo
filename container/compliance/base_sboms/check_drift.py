# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Detect drift in tracked base-image SBOMs.

Reads container/compliance/base_sboms/manifest.json and, for each entry,
resolves the current registry manifest-list digest for image:tag. Fails
if any digest differs from what's recorded — that means the underlying
base image was rebuilt without our SBOM corpus being refreshed, so the
SBOM-diff verifier would be checking against the wrong reference.

Replaces the planned weekly cron: rather than always re-pulling on a
schedule (most refresh runs would be no-ops), we make every CI run
verify the digests it cares about. The first run after a base bump
fails fast and points the human at the refresh path.

Resolution uses `docker buildx imagetools inspect --raw <ref>` and
hashes the returned canonical manifest bytes. By OCI spec, sha256 of
the raw manifest IS its digest, so this is deterministic and doesn't
depend on a Go-template format string (which renders Digest as a
struct dump on multi-arch manifests). Available on any runner with
buildx (every runner we use to build images). No registry credentials
needed for public bases; private bases must be `docker login`-ed by
the calling workflow.

Exit codes:
  0  all manifest entries match the live registry, OR manifest is empty
  1  one or more entries drifted (output explains which and how to fix)
  2  registry/network failure on at least one entry (treated as drift —
     CI should not silently pass if we can't verify)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.json"


def resolve_current_digest(image: str, tag: str) -> str:
    """Return the registry's current manifest-list digest for image:tag.

    Per the OCI distribution spec, the manifest's digest is the SHA-256
    of its canonical-form bytes. `imagetools inspect --raw` returns
    exactly those bytes, so we can hash them locally and avoid the
    template-formatting issues with imagetools' default output.

    Raises subprocess.CalledProcessError on registry/network failure;
    the caller folds that into the drift report rather than crashing.
    """
    ref = f"{image}:{tag}"
    result = subprocess.run(
        ["docker", "buildx", "imagetools", "inspect", "--raw", ref],
        check=True,
        capture_output=True,
    )
    return "sha256:" + hashlib.sha256(result.stdout).hexdigest()


def check(manifest_path: Path) -> int:
    if not manifest_path.is_file():
        logger.error("manifest.json not found at %s", manifest_path)
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("entries", []) or []
    if not entries:
        print("base_sboms manifest is empty; nothing to verify.")
        return 0

    drifts: list[str] = []
    network_fails: list[str] = []

    for entry in entries:
        image = entry.get("image")
        tag = entry.get("tag")
        recorded = entry.get("digest")
        sbom = entry.get("sbom", "<missing>")

        if not image or not tag or not recorded:
            drifts.append(
                f"manifest entry malformed (missing image/tag/digest): {entry!r}"
            )
            continue

        try:
            current = resolve_current_digest(image, tag)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            network_fails.append(f"{image}:{tag}: registry lookup failed ({stderr})")
            continue

        if current != recorded:
            drifts.append(
                f"{image}:{tag}: DRIFT\n"
                f"      recorded: {recorded}\n"
                f"      current : {current}\n"
                f"      action  : refresh base_sboms/{sbom} "
                f"(run container/compliance/base_sboms/refresh.py) "
                f"and update manifest.json"
            )
        else:
            print(f"  {image}:{tag} OK ({recorded[:19]}…)")

    if drifts or network_fails:
        print("", file=sys.stderr)
        if drifts:
            print(
                f"Base-image drift detected ({len(drifts)} entr"
                f"{'y' if len(drifts) == 1 else 'ies'}):",
                file=sys.stderr,
            )
            for d in drifts:
                print(f"  - {d}", file=sys.stderr)
        if network_fails:
            print(
                f"Registry lookup failures ({len(network_fails)} entr"
                f"{'y' if len(network_fails) == 1 else 'ies'}; "
                "treated as drift):",
                file=sys.stderr,
            )
            for f in network_fails:
                print(f"  - {f}", file=sys.stderr)
        return 1 if drifts and not network_fails else 2

    print(f"All {len(entries)} tracked base images current.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_drift",
        description="Verify base-image digests in base_sboms/manifest.json against the live registry.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_MANIFEST_PATH,
        help="Path to manifest.json (default: %(default)s)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s [%(name)s]: %(message)s",
    )

    return check(args.manifest)


if __name__ == "__main__":
    sys.exit(main())
