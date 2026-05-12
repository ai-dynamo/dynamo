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

Final output: /sources.zip, packed with deterministic ordering and a
fixed mtime so the artifact's sha256 is stable across rebuilds. The
companion OSRB bundle (osrb/package.py) records this archive's
sha256 in build-provenance.json for cross-verification.

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
import contextlib
import json
import logging
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)


def _enumerate_installed_dpkgs() -> set[str]:
    """Return the set of installed dpkg package names."""
    result = subprocess.run(
        ["dpkg-query", "-W", "-f=${Package}\\n"],
        check=True, capture_output=True, text=True,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _baseline_dpkg_names(baseline_sbom: Path) -> set[str]:
    """Extract dpkg (name) tuples from a slim CycloneDX baseline SBOM.

    Matches by name only — source-package versions diverge from binary-
    package versions in Debian/Ubuntu (a single source package can produce
    several binary versions; security updates rev the binary but the
    upstream source is the same). Filtering by name gives us "what NGC's
    baseline owns at the source-package level."
    """
    doc = json.loads(baseline_sbom.read_text(encoding="utf-8"))
    out: set[str] = set()
    for c in doc.get("components", []) or []:
        purl = c.get("purl") or ""
        if not purl.startswith("pkg:deb/"):
            continue
        name = c.get("name")
        if name:
            out.add(name)
    return out


@contextlib.contextmanager
def _deb_src_enabled():
    """Temporarily enable `deb-src` lines in /etc/apt/sources.list*.

    Ubuntu's default cuda-dl-base apt config omits deb-src; `apt-get
    source` needs them. Toggle on, `apt-get update`, do the work,
    restore the originals so the runtime image's apt state is unchanged
    after the sources stage exits.
    """
    sources_paths: list[Path] = [Path("/etc/apt/sources.list")]
    sources_paths.extend(Path("/etc/apt/sources.list.d").glob("*.list"))
    sources_paths.extend(Path("/etc/apt/sources.list.d").glob("*.sources"))
    backups: dict[Path, bytes] = {}
    try:
        for p in sources_paths:
            if not p.is_file():
                continue
            data = p.read_bytes()
            backups[p] = data
            text = data.decode("utf-8", errors="replace")
            new = "\n".join(
                # Uncomment `# deb-src` lines AND mirror `deb` → `deb-src`
                # for lines we haven't already enabled.
                _maybe_enable_src(line) for line in text.splitlines()
            )
            p.write_text(new, encoding="utf-8")
        subprocess.run(["apt-get", "update"], check=True)
        yield
    finally:
        for p, data in backups.items():
            p.write_bytes(data)


def _maybe_enable_src(line: str) -> str:
    """Toggle a single sources.list line so deb-src is enabled.

    Cases:
      `# deb-src https://...`  →  uncomment
      `deb https://...`        →  emit as-is, also append a sibling deb-src
      anything else            →  unchanged
    """
    stripped = line.lstrip()
    if stripped.startswith("# deb-src") or stripped.startswith("#deb-src"):
        # uncomment
        idx = line.find("#")
        return line[:idx] + line[idx + 1 :].lstrip()
    if stripped.startswith("deb ") or stripped.startswith("deb\t"):
        # emit original AND a deb-src variant on the next line
        src_variant = line.replace("deb ", "deb-src ", 1) if "deb " in line else line.replace("deb\t", "deb-src\t", 1)
        return f"{line}\n{src_variant}"
    return line


def collect_dpkg_sources(baseline_sbom: Path | None, output_dir: Path) -> int:
    """Diff installed dpkg state against the baseline, fetch source for the deltas.

    Returns the number of packages whose source was successfully fetched.

    For each delta package, `apt-get source --download-only -d` fetches
    the `.dsc` + `.tar.{xz,gz}` and (when present) `.debian.tar.{xz,gz}`
    into the cwd. NVIDIA-proprietary packages from the cuda repos don't
    have public source — we log and continue rather than failing the
    build, matching how Debian's `non-free` repository handles the
    same case.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        installed = _enumerate_installed_dpkgs()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.error("dpkg-query failed (is dpkg installed?): %s", exc)
        return 0
    logger.info("Installed dpkg packages: %d", len(installed))

    if baseline_sbom is None:
        delta_names = installed
        logger.info(
            "No baseline configured; will attempt source for every installed "
            "package (%d total). This is intentional fallback for unconfigured "
            "builds; usually a baseline should be specified.",
            len(delta_names),
        )
    else:
        baseline_names = _baseline_dpkg_names(baseline_sbom)
        delta_names = installed - baseline_names
        logger.info(
            "Baseline owns %d dpkg packages; delta = %d packages to fetch source for.",
            len(baseline_names),
            len(delta_names),
        )

    if not delta_names:
        return 0

    fetched = 0
    skipped: list[str] = []
    with _deb_src_enabled():
        for name in sorted(delta_names):
            try:
                subprocess.run(
                    ["apt-get", "source", "--only-source", "--download-only", name],
                    check=True,
                    cwd=output_dir,
                    env={**os.environ, "DEBIAN_FRONTEND": "noninteractive"},
                    capture_output=True,
                )
                fetched += 1
            except subprocess.CalledProcessError:
                # Most common cause: NVIDIA-proprietary repos don't publish
                # source. Documented in the bundle README. Log so an auditor
                # can see which packages were skipped and why.
                skipped.append(name)
                logger.debug("no public source for %s; skipping", name)

    if skipped:
        logger.warning(
            "Skipped %d dpkg packages with no public source repo "
            "(typically NVIDIA-proprietary; see bundle README): %s",
            len(skipped),
            ", ".join(skipped[:20]) + (" …" if len(skipped) > 20 else ""),
        )
    logger.info("dpkg sources collected: %d / %d", fetched, len(delta_names))
    return fetched


# First-party Rust crate prefixes. Crates whose name starts with any of
# these are NVIDIA-authored — source lives on GitHub, not redistribution
# of someone else's OSS, so we don't ship it in the OSRB sources archive.
_FIRST_PARTY_RUST_PREFIXES = ("dynamo-", "kvbm-", "nixl-")


def _shipped_rust_crates(venv_root: Path) -> set[tuple[str, str]]:
    """Walk every installed wheel's embedded CycloneDX SBOM and return the
    set of (name, version) tuples for Rust crates that ship in this image.

    Mirrors the path generators/rust.py uses to discover wheel-bundled SBOMs:
    <venv>/lib/python*/site-packages/*.dist-info/sboms/*.cyclonedx.json.
    """
    crates: set[tuple[str, str]] = set()
    sbom_paths: list[Path] = []
    for site in venv_root.glob("lib/python*/site-packages"):
        sbom_paths.extend(site.glob("*.dist-info/sboms/*.cyclonedx.json"))
    for sbom in sbom_paths:
        try:
            doc = json.loads(sbom.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("could not read Rust SBOM at %s", sbom)
            continue
        for c in doc.get("components", []) or []:
            purl = c.get("purl") or ""
            if not purl.startswith("pkg:cargo/"):
                continue
            name = c.get("name")
            version = c.get("version")
            if name and version:
                crates.add((name, str(version)))
    logger.info(
        "Found %d distinct Rust crates across %d wheel SBOMs in %s",
        len(crates),
        len(sbom_paths),
        venv_root,
    )
    return crates


def collect_rust_sources(
    venv_root: Path, vendor_full: Path, output_dir: Path
) -> int:
    """Copy third-party Rust crate sources into output_dir.

    Walks installed wheels' embedded SBOMs to discover what shipped,
    then for each (name, version) copies vendor_full/<name>-<version>/
    to output_dir/vendor/<name>-<version>/ EXCEPT for first-party
    crates (dynamo-*, kvbm-*, nixl-*), which are NVIDIA-authored.

    Cargo.toml + Cargo.lock from the workspace are copied alongside so a
    consumer can reconstruct a buildable vendor tree.

    Returns the number of crate directories copied.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    vendor_dest = output_dir / "vendor"
    vendor_dest.mkdir(parents=True, exist_ok=True)

    if not vendor_full.is_dir():
        logger.warning(
            "Rust vendor tree not found at %s. Did wheel_builder run "
            "cargo vendor with ENABLE_SOURCE_ARCHIVAL=true?",
            vendor_full,
        )
        return 0

    crates = _shipped_rust_crates(venv_root)
    copied = 0
    skipped_first_party = 0
    missing_in_vendor: list[str] = []
    for (name, version) in sorted(crates):
        if name.startswith(_FIRST_PARTY_RUST_PREFIXES):
            skipped_first_party += 1
            continue
        crate_dir = vendor_full / f"{name}-{version}"
        if not crate_dir.is_dir():
            missing_in_vendor.append(f"{name}-{version}")
            continue
        shutil.copytree(crate_dir, vendor_dest / f"{name}-{version}")
        copied += 1

    # Copy workspace manifest context so the vendor tree is buildable.
    for manifest_name in ("Cargo.toml", "Cargo.lock"):
        candidate = vendor_full / manifest_name
        if candidate.is_file():
            shutil.copy2(candidate, output_dir / manifest_name)

    logger.info(
        "Rust sources collected: %d third-party crates (skipped %d "
        "first-party; %d crates declared in SBOMs but missing from "
        "vendor tree)",
        copied,
        skipped_first_party,
        len(missing_in_vendor),
    )
    if missing_in_vendor and len(missing_in_vendor) <= 20:
        logger.debug("missing-from-vendor: %s", ", ".join(missing_in_vendor))
    return copied


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


def pack_sources_zip(sources_root: Path, zip_path: Path) -> None:
    """Pack sources_root as a deterministic-ish zip.

    Walks in sorted order with a fixed mtime per ZipInfo — central-directory
    layout makes byte-exact reproducibility imperfect for zip vs tar, but
    these knobs are enough for OSRB cross-verification: the bundle records
    the sha256 of this archive in build-provenance.json.
    """
    if not sources_root.is_dir():
        raise FileNotFoundError(f"sources root missing: {sources_root}")
    fixed_mtime = (1980, 1, 1, 0, 0, 0)
    paths = sorted(p for p in sources_root.rglob("*") if p.is_file())
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in paths:
            arcname = path.relative_to(sources_root.parent).as_posix()
            info = zipfile.ZipInfo(filename=arcname, date_time=fixed_mtime)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with path.open("rb") as f:
                zf.writestr(info, f.read())
    logger.info("Wrote %s (%.1f MB, %d files)",
                zip_path, zip_path.stat().st_size / 1e6, len(paths))


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
        "--output-zip",
        type=Path,
        default=Path("/sources.zip"),
        help="Where to write the final sources zip.",
    )
    parser.add_argument(
        "--sources-root",
        type=Path,
        default=Path("/sources"),
        help="Working directory for collected sources.",
    )
    parser.add_argument(
        "--baseline-sbom",
        type=Path,
        default=None,
        help=(
            "Path to the slim CycloneDX baseline SBOM whose components we DON'T "
            "ship source for (the upstream base owns redistribution for them). "
            "Same file the runtime template's licenses stage subtracts at "
            "NOTICES time. Pass via "
            "`${BASELINE_SBOM_FILE:+--baseline-sbom /opt/compliance/base_sboms/${BASELINE_SBOM_FILE}}` "
            "from the sources_collect Dockerfile stage; omit when no baseline "
            "is configured (the dpkg collector then falls back to shipping "
            "source for every installed package)."
        ),
    )
    parser.add_argument(
        "--native-source-dir",
        type=Path,
        default=Path("/opt/native-sources"),
        help="Where the Dockerfile COPY'd preserved native source archives.",
    )
    parser.add_argument(
        "--rust-venv",
        type=Path,
        default=Path("/opt/dynamo/venv"),
        help=(
            "Runtime venv whose installed wheels carry CycloneDX SBOMs. "
            "Walked for embedded *.dist-info/sboms/*.cyclonedx.json files "
            "to discover the third-party crate set we ship."
        ),
    )
    parser.add_argument(
        "--rust-vendor-full",
        type=Path,
        default=Path("/opt/dynamo-vendor-full"),
        help=(
            "Workspace cargo-vendor tree produced by wheel_builder when "
            "ENABLE_SOURCE_ARCHIVAL=true. Filtered against the rust-venv's "
            "shipped-crates set to produce the third-party-only output."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s [%(name)s]: %(message)s",
    )

    ecosystems = args.ecosystem or ["dpkg", "native"]

    args.sources_root.mkdir(parents=True, exist_ok=True)
    base_sbom: Path | None = args.baseline_sbom
    if "dpkg" in ecosystems:
        if base_sbom is None:
            logger.warning(
                "No --baseline-sbom passed; dpkg source collection will fetch sources for "
                "the entire installed package set rather than just additions on top of the base."
            )
        elif not base_sbom.is_file():
            logger.warning(
                "--baseline-sbom %s does not exist; falling back to ship-everything dpkg mode.",
                base_sbom,
            )
            base_sbom = None

    counts: dict[str, int] = {}
    if "dpkg" in ecosystems:
        counts["dpkg"] = collect_dpkg_sources(base_sbom, args.sources_root / "dpkg")
    if "rust" in ecosystems:
        counts["rust"] = collect_rust_sources(
            args.rust_venv, args.rust_vendor_full, args.sources_root / "rust"
        )
    if "native" in ecosystems:
        counts["native"] = collect_native_sources(
            args.native_source_dir, args.sources_root / "native"
        )

    pack_sources_zip(args.sources_root, args.output_zip)
    logger.info("Source archival complete: %s", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
