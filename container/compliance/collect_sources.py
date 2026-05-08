#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source archival orchestrator.

Run inside the per-image `sources` Dockerfile stage. Collects source
archives for everything we ship on top of the FROM base, suitable for
OSRB submission and GPL/LGPL distribution-on-request compliance.

Per-ecosystem strategy (matches the plan in
~/.claude/plans/ok-i-think-this-parallel-widget.md):

  dpkg     diff against the base-SBOM to identify packages we ADDED
           (vs. inherited from cuda-dl-base / NGC base), then run
           `apt-get source --only-source --download-only -d <pkg>`
           on each delta. Skip silently on packages that have no
           source repo (proprietary CUDA, NVIDIA-internal, etc.) —
           those are documented in license_overrides.yaml as
           "source not available; see EULA".

  rust     filtered vendor tree from the wheel_builder stage
           (cargo vendor --locked, then filter to SBOM-declared
           components). The full vendor is produced upstream in
           wheel_builder; this script just COPYs in the filtered
           subset and tars it.

  go       go mod vendor per Go module. For the dynamo runtime
           templates this is empty (no Go binaries); operator /
           snapshot-agent / EPP have it.

  native   preserve source tarballs for from-source components
           (criu, ucx, libfabric, ffmpeg, gdrcopy, NIXL, etc.).
           These were downloaded by the wheel_builder stage and
           are COPYd in by the Dockerfile.

  python   skipped intentionally. Source already ships in the
           installed wheels (sdists / .py source files in
           site-packages); pip download --no-binary would
           duplicate ~GB for zero compliance value.

Final output: /sources.tar.gz, packed with reproducible-tar flags so
the artifact's hash is deterministic across rebuilds.

Runs in the post-merge / RC / release CI pipelines only — skipped on
PR builds (storage cost too high per-build, and PR doesn't change
the source-of-truth a release ships from).

TODO: implement the dpkg diff-and-fetch logic. Skeleton in place so
the Dockerfile stage and CI integration can be built and tested. Real
implementation lands when the corresponding Dockerfile stage is wired
up to expose /var/cache/apt with deb-src configured.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _resolve_base_sbom(manifest_path: Path, framework: str, target: str, cuda_version: str) -> Path | None:
    """Look up the base SBOM for this image's FROM in base_sboms/manifest.json."""
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("entries", []) or []
    if not entries:
        return None
    # TODO: refine matching once manifest.json carries explicit
    # (framework, target, cuda) keys. For now, return the first entry.
    sbom_name = entries[0].get("sbom")
    if not sbom_name:
        return None
    return manifest_path.parent / sbom_name


def collect_dpkg_sources(base_sbom: Path | None, output_dir: Path) -> int:
    """Diff installed dpkg state against base SBOM, fetch source for the deltas.

    Returns the number of packages whose source was successfully fetched.

    TODO: implement. Skeleton:
      1. Run `dpkg-query -W -f='${Package}\t${Version}\n'` to enumerate.
      2. If base_sbom: load it, extract `pkg:deb/...` (name, version) tuples.
      3. delta = installed - base.
      4. For each delta package:
           apt-get source --only-source --download-only -d <pkg>
           Move resulting *.dsc / *.tar.{xz,gz} into `output_dir`.
           Skip silently on apt errors (proprietary repos with no source).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.warning(
        "dpkg source collection not yet implemented. "
        "Output will be empty until the dpkg pipeline lands."
    )
    return 0


def collect_native_sources(workspace_native_dir: Path, output_dir: Path) -> int:
    """Copy native source tarballs preserved by builder stages.

    Each builder Dockerfile that does `RUN git clone …` or `wget …tar` should
    preserve the resulting archive at /tmp/native-sources/<name>-<version>.tar.gz
    (or similar). The Dockerfile's `sources` stage `COPY --from=builder
    /tmp/native-sources /opt/native-sources` puts them where this script can find them.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if not workspace_native_dir.is_dir():
        logger.warning("No native source directory at %s; skipping.", workspace_native_dir)
        return 0
    n = 0
    for item in workspace_native_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, output_dir / item.name)
            n += 1
    logger.info("Copied %d native source archive(s) from %s", n, workspace_native_dir)
    return n


def pack_sources_tarball(sources_root: Path, tarball_path: Path) -> None:
    """Create a reproducible tarball of sources_root.

    Reproducible flags so the hash is deterministic across rebuilds — important
    for OSRB integrity checks and our own provenance.
    """
    if not sources_root.is_dir():
        raise FileNotFoundError(f"sources root missing: {sources_root}")
    cmd = [
        "tar",
        "--owner=0",
        "--group=0",
        "--mtime=1980-01-01",
        "--sort=name",
        "-czf",
        str(tarball_path),
        "-C",
        str(sources_root.parent),
        sources_root.name,
    ]
    subprocess.run(cmd, check=True)
    logger.info("Wrote %s (%.1f MB)", tarball_path, tarball_path.stat().st_size / 1e6)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ecosystem",
        action="append",
        default=[],
        help="Ecosystems to collect (repeatable). Valid: dpkg, rust, go, native. "
        "Default: all applicable.",
    )
    parser.add_argument(
        "--output-tarball",
        type=Path,
        default=Path("/sources.tar.gz"),
        help="Where to write the final reproducible tarball.",
    )
    parser.add_argument(
        "--sources-root",
        type=Path,
        default=Path("/sources"),
        help="Working directory for collected sources.",
    )
    parser.add_argument(
        "--base-sbom-manifest",
        type=Path,
        default=Path("/opt/compliance/base_sboms/manifest.json"),
        help="Path to the base-SBOM manifest (used by dpkg diff-against-base).",
    )
    parser.add_argument(
        "--framework",
        default="",
        help="Framework name (used by base-SBOM lookup).",
    )
    parser.add_argument(
        "--target",
        default="",
        help="Image target (runtime/frontend/planner; used by base-SBOM lookup).",
    )
    parser.add_argument(
        "--cuda-version",
        default="",
        help="CUDA version (used by base-SBOM lookup).",
    )
    parser.add_argument(
        "--native-source-dir",
        type=Path,
        default=Path("/opt/native-sources"),
        help="Where the Dockerfile COPY'd preserved native source archives.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s [%(name)s]: %(message)s",
    )

    ecosystems = args.ecosystem or ["dpkg", "native"]

    args.sources_root.mkdir(parents=True, exist_ok=True)
    base_sbom: Path | None = None
    if "dpkg" in ecosystems:
        base_sbom = _resolve_base_sbom(
            args.base_sbom_manifest, args.framework, args.target, args.cuda_version
        )
        if base_sbom is None:
            logger.warning(
                "No base-SBOM resolved; dpkg source collection will fetch sources for "
                "the entire installed package set rather than just additions on top of the base."
            )

    counts: dict[str, int] = {}
    if "dpkg" in ecosystems:
        counts["dpkg"] = collect_dpkg_sources(base_sbom, args.sources_root / "dpkg")
    if "native" in ecosystems:
        counts["native"] = collect_native_sources(
            args.native_source_dir, args.sources_root / "native"
        )

    pack_sources_tarball(args.sources_root, args.output_tarball)
    logger.info("Source archival complete: %s", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
