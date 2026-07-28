#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source archival orchestrator.

Run inside the per-image `sources` Dockerfile stage. Collects source
archives for everything we ship on top of the FROM base, suitable for
OSRB submission and GPL/LGPL distribution-on-request compliance.

Per-ecosystem strategy (matches the plan in
~/.claude/plans/ok-i-think-this-parallel-widget.md):

  dpkg     diff against the base-SBOM to identify binary packages we
           ADDED (vs. inherited from cuda-dl-base / NGC base), resolve
           each to its source package, then run `apt-get source
           --only-source --download-only -d <src>` once per source.
           Packages with no public source repo (proprietary CUDA,
           NVIDIA-internal, etc.) are recorded in manifest.json with a
           reason; they are also documented in license_overrides.yaml
           as "source not available; see EULA".

  rust     filtered vendor tree from the wheel_builder stage
           (cargo vendor --locked, then filter to SBOM-declared
           components). The full vendor is produced upstream in
           wheel_builder; this script just COPYs in the filtered
           subset and tars it.

  go       go mod vendor per Go module. For the dynamo runtime
           templates this is empty (no Go binaries); operator /
           snapshot-agent / EPP have it.

  native   preserve source tarballs for from-source components
           (criu, ucx, libfabric, ffmpeg, gdrcopy, etc.). These were
           downloaded by the wheel_builder stage and are COPYd in by
           the Dockerfile.

  python   skipped intentionally. Source already ships in the
           installed wheels (sdists / .py source files in
           site-packages); pip download --no-binary would
           duplicate ~GB for zero compliance value.

Final output: /sources.tar.gz, packed deterministically (sorted
members, pinned mtimes, neutral ownership) so the archive's sha256 is
stable across rebuilds. It carries README.md, manifest.json
(per-ecosystem declared/collected/missing) and CHECKSUMS.sha256
alongside the sources. The companion OSRB bundle (osrb/package.py)
records this archive's sha256 in build-provenance.json for
cross-verification.

The tree is packed rather than exported file-by-file because it runs
to ~1 GB per architecture: exporting it unpacked puts that, plus tens
of thousands of files, on the CI runner's local disk twice over.

Gated on ENABLE_SOURCE_ARCHIVAL; callers opt in (nightly / RC /
release) because the tree runs to hundreds of MB per image.

Collection is best-effort by design: a gap never fails the build, but
every gap is counted and named in manifest.json so an auditor can see
exactly what is and isn't covered.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

logger = logging.getLogger(__name__)


def _enumerate_installed_dpkgs(root: Path = Path("/")) -> set[str]:
    """Return the set of installed dpkg package names."""
    cmd = ["dpkg-query", "-W", "-f=${Package}\\n"]
    if root != Path("/"):
        cmd.insert(1, f"--admindir={root / 'var/lib/dpkg'}")
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _binary_to_source_map(root: Path = Path("/")) -> dict[str, str]:
    """Map each installed binary package to the source package it builds from.

    `apt-get source` resolves *source* package names, but dpkg-query reports
    *binary* names, and the two frequently differ — udev, libsystemd-shared
    and systemd-dev all come from the `systemd` source. dpkg's
    ${source:Package} field carries the mapping directly and falls back to
    the binary name when a package declares no distinct source.
    """
    cmd = ["dpkg-query", "-W", "-f=${Package}\\t${source:Package}\\n"]
    if root != Path("/"):
        cmd.insert(1, f"--admindir={root / 'var/lib/dpkg'}")
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    mapping: dict[str, str] = {}
    for line in result.stdout.splitlines():
        binary, _, source = line.partition("\t")
        binary, source = binary.strip(), source.strip()
        if binary:
            mapping[binary] = source or binary
    return mapping


# Vendor repositories that publish no source. Matched as prefixes against the
# source-package name so a genuine proprietary skip stays distinguishable from
# a fetch that simply failed.
_NO_PUBLIC_SOURCE_PREFIXES = (
    "cuda-",
    "doca-",
    "libcudnn",
    "libnccl",
    "mlnx-",
    "nsight-",
    "nvidia-",
)


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

    Two on-disk formats coexist:
      - Legacy `*.list` files (one `deb …` directive per line). Ubuntu
        pre-noble images, cuda-dl-base's NVIDIA repo files, etc.
      - Deb822 `*.sources` files (paragraph format with `Types:`,
        `URIs:`, `Suites:`, `Components:`). Ubuntu 24.04's default
        `/etc/apt/sources.list.d/ubuntu.sources`.

    Dispatch on filename suffix; the two formats need different rewrites.
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
            if p.suffix == ".sources":
                new = _rewrite_deb822(text)
            else:
                new = "\n".join(_maybe_enable_src(line) for line in text.splitlines())
            p.write_text(new, encoding="utf-8")
        subprocess.run(["apt-get", "update"], check=True)
        yield
    finally:
        for p, data in backups.items():
            p.write_bytes(data)


def _maybe_enable_src(line: str) -> str:
    """Toggle a single legacy `*.list` line so deb-src is enabled.

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
        src_variant = (
            line.replace("deb ", "deb-src ", 1)
            if "deb " in line
            else line.replace("deb\t", "deb-src\t", 1)
        )
        return f"{line}\n{src_variant}"
    return line


def _rewrite_deb822(text: str) -> str:
    """Add `deb-src` to every `Types:` field in a deb822 sources file.

    Ubuntu 24.04 ships its apt config as deb822 paragraphs:

        Types: deb
        URIs: http://archive.ubuntu.com/ubuntu
        Suites: noble noble-updates noble-backports
        Components: main restricted universe multiverse

    `apt-get source` needs `Types: deb deb-src` (or a separate `deb-src`
    paragraph) to find source packages. We append `deb-src` to the
    existing Types: line so a single paragraph covers both — minimal
    edit, no paragraph duplication. Idempotent: re-running on
    already-rewritten content leaves it unchanged.
    """
    out_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        # Field names are case-insensitive per deb822 spec.
        if stripped[:6].lower() == "types:":
            indent = line[: len(line) - len(stripped)]
            _, _, value = stripped.partition(":")
            tokens = value.split()
            if "deb" in tokens and "deb-src" not in tokens:
                tokens.append("deb-src")
                out_lines.append(f"{indent}Types: {' '.join(tokens)}")
                continue
        out_lines.append(line)
    return "\n".join(out_lines)


def collect_dpkg_sources(
    baseline_sbom: Path | None, output_dir: Path, dpkg_root: Path = Path("/")
) -> dict:
    """Diff installed dpkg state against the baseline, fetch source for the deltas.

    The baseline diff runs in binary-package space (that is what the baseline
    SBOM enumerates), then the delta is collapsed onto source packages before
    fetching, so a source package that produced several installed binaries is
    fetched once rather than once per binary.

    For each delta package, `apt-get source --download-only -d` fetches
    the `.dsc` + `.tar.{xz,gz}` and (when present) `.debian.tar.{xz,gz}`
    into the cwd. NVIDIA-proprietary packages from the cuda repos don't
    have public source — we log and continue rather than failing the
    build, matching how Debian's `non-free` repository handles the
    same case.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        installed = _enumerate_installed_dpkgs(dpkg_root)
        bin_to_src = _binary_to_source_map(dpkg_root)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.error("dpkg-query failed (is dpkg installed?): %s", exc)
        return {"declared": 0, "collected": 0, "missing": [], "skipped": []}
    logger.info("Installed dpkg packages under %s: %d", dpkg_root, len(installed))

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
        return {"declared": 0, "collected": 0, "missing": [], "skipped": []}

    # Collapse the binary delta onto source packages: systemd, udev,
    # libsystemd-shared and systemd-dev are one `systemd` fetch, not four.
    source_targets: dict[str, set[str]] = {}
    for binary in delta_names:
        source_targets.setdefault(bin_to_src.get(binary, binary), set()).add(binary)
    logger.info(
        "Delta of %d binary packages resolves to %d source packages.",
        len(delta_names),
        len(source_targets),
    )

    fetched = 0
    skipped: list[dict] = []
    with _deb_src_enabled():
        for name in sorted(source_targets):
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
                proprietary = name.startswith(_NO_PUBLIC_SOURCE_PREFIXES)
                skipped.append(
                    {
                        "source_package": name,
                        "binary_packages": sorted(source_targets[name]),
                        "reason": (
                            "no public source repo (vendor-proprietary)"
                            if proprietary
                            else "apt-get source failed"
                        ),
                    }
                )

    unexplained = [s for s in skipped if s["reason"] == "apt-get source failed"]
    if skipped:
        logger.warning(
            "Skipped %d dpkg source packages (%d vendor-proprietary, "
            "%d fetch failures): %s",
            len(skipped),
            len(skipped) - len(unexplained),
            len(unexplained),
            ", ".join(s["source_package"] for s in skipped[:20])
            + (" …" if len(skipped) > 20 else ""),
        )
    if unexplained:
        logger.warning(
            "dpkg source packages that should have public source but could not "
            "be fetched: %s",
            ", ".join(s["source_package"] for s in unexplained),
        )
    logger.info("dpkg sources collected: %d / %d", fetched, len(source_targets))
    return {
        "declared": len(source_targets),
        "collected": fetched,
        "missing": [],
        "skipped": skipped,
    }


# First-party Rust crate prefixes. Crates whose name starts with any of
# these are NVIDIA-authored — source lives on GitHub, not redistribution
# of someone else's OSS, so we don't ship it in the OSRB sources archive.
_FIRST_PARTY_RUST_PREFIXES = ("dynamo-", "kvbm-", "nixl-")


def _shipped_rust_crates(site_packages_dirs: list[Path]) -> set[tuple[str, str]]:
    """Walk every installed wheel's embedded CycloneDX SBOM and return the
    set of (name, version) tuples for Rust crates that ship in this image.

    Mirrors generators/rust.py's discovery path. Accepts a list of
    site-packages directories rather than a venv root because runtimes that
    `pip install --break-system-packages` (sglang) ship packages under
    `/usr/lib/python3/dist-packages` instead of the `lib/python*/site-packages`
    layout a venv produces.
    """
    crates: set[tuple[str, str]] = set()
    sbom_paths: list[Path] = []
    for site in site_packages_dirs:
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
        "Found %d distinct Rust crates across %d wheel SBOMs under %s",
        len(crates),
        len(sbom_paths),
        ", ".join(str(d) for d in site_packages_dirs),
    )
    return crates


def _crate_id(crate_dir: Path) -> tuple[str, str] | None:
    """Read (name, version) out of a vendored crate's own Cargo.toml."""
    try:
        text = (crate_dir / "Cargo.toml").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    name = version = None
    try:
        import tomllib

        package = tomllib.loads(text).get("package") or {}
        name, version = package.get("name"), package.get("version")
    except Exception:
        # tomllib is 3.11+, and a handful of published manifests carry syntax
        # it rejects. [package] is the first table cargo emits, so scan it.
        match = re.search(r"^\s*\[package\]\s*$(.*?)(?=^\s*\[)", text, re.M | re.S)
        block = match.group(1) if match else text
        name_match = re.search(r'^\s*name\s*=\s*"([^"]+)"', block, re.M)
        version_match = re.search(r'^\s*version\s*=\s*"([^"]+)"', block, re.M)
        name = name_match.group(1) if name_match else None
        version = version_match.group(1) if version_match else None

    return (name, str(version)) if name and version else None


def _index_vendor_tree(vendor_full: Path) -> dict[tuple[str, str], Path]:
    """Map (crate name, version) to its directory in a cargo-vendor tree.

    Directory names cannot be derived from the crate identity: `cargo vendor`
    names a directory after the crate alone when a single version is
    vendored, and `<name>-<version>` only for the additional versions of a
    crate that appears more than once. Reading each manifest recovers the
    pair regardless of which form cargo chose.
    """
    index: dict[tuple[str, str], Path] = {}
    for entry in sorted(vendor_full.iterdir()):
        if not entry.is_dir():
            continue
        crate_id = _crate_id(entry)
        if crate_id is not None:
            index[crate_id] = entry
    logger.info("Indexed %d crates in vendor tree %s", len(index), vendor_full)
    return index


def collect_rust_sources(
    site_packages_dirs: list[Path], vendor_full: Path, output_dir: Path
) -> dict:
    """Copy third-party Rust crate sources into output_dir.

    Walks installed wheels' embedded SBOMs to discover what shipped, then
    copies each declared crate out of the indexed vendor tree into
    output_dir/vendor/<name>-<version>/ EXCEPT for first-party crates
    (dynamo-*, kvbm-*, nixl-*), which are NVIDIA-authored. The output is
    always named <name>-<version> even where the vendor tree used the bare
    crate name, so the archive is unambiguous for an auditor.

    Cargo.toml + Cargo.lock from the workspace are copied alongside so a
    consumer can reconstruct a buildable vendor tree.
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
        return {"declared": 0, "collected": 0, "missing": [], "skipped": []}

    index = _index_vendor_tree(vendor_full)
    crates = _shipped_rust_crates(site_packages_dirs)
    copied = 0
    skipped_first_party = 0
    missing_in_vendor: list[str] = []
    for name, version in sorted(crates):
        if name.startswith(_FIRST_PARTY_RUST_PREFIXES):
            skipped_first_party += 1
            continue
        crate_dir = index.get((name, version))
        if crate_dir is None:
            missing_in_vendor.append(f"{name}-{version}")
            continue
        shutil.copytree(crate_dir, vendor_dest / f"{name}-{version}")
        copied += 1

    # Copy workspace manifest context so the vendor tree is buildable.
    for manifest_name in ("Cargo.toml", "Cargo.lock"):
        candidate = vendor_full / manifest_name
        if candidate.is_file():
            shutil.copy2(candidate, output_dir / manifest_name)

    declared = len(crates) - skipped_first_party
    logger.info(
        "Rust sources collected: %d / %d third-party crates (skipped %d first-party)",
        copied,
        declared,
        skipped_first_party,
    )
    if missing_in_vendor:
        logger.warning(
            "%d crates declared in wheel SBOMs are missing from the vendor "
            "tree and are NOT in the archive: %s",
            len(missing_in_vendor),
            ", ".join(sorted(missing_in_vendor)[:20])
            + (" …" if len(missing_in_vendor) > 20 else ""),
        )
    return {
        "declared": declared,
        "collected": copied,
        "missing": sorted(missing_in_vendor),
        "skipped": [],
        "first_party_excluded": skipped_first_party,
    }


# First-party Go module prefixes. Our own code; source public on GitHub.
_FIRST_PARTY_GO_PREFIXES = ("github.com/ai-dynamo/",)


def collect_go_sources(go_vendor_dir: Path, output_dir: Path) -> dict:
    """Copy third-party Go module sources from a `go mod vendor` tree.

    The simplest correct approach: copy the entire vendor tree, then
    delete any first-party subtrees. Avoids fragile module-root
    detection heuristics — `go mod vendor` already produces a clean
    directory structure where module paths map directly to filesystem
    paths.

    Operator + snapshot don't sit on a baseline that contains Go
    modules (distroless/go and cuda-dl-base ship Go binaries, not
    module sources), so the first-party filter is the only filter
    needed.

    Module counts are approximate — a `go mod vendor` tree carries no
    declared-set to reconcile against, so declared == collected here.
    """
    if not go_vendor_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.warning(
            "Go vendor tree not found at %s. Did the Go builder run "
            "`go mod vendor` with ENABLE_SOURCE_ARCHIVAL=true?",
            go_vendor_dir,
        )
        return {"declared": 0, "collected": 0, "missing": [], "skipped": []}

    # shutil.copytree requires dirs_exist_ok=True to merge into an
    # existing output directory; we always create the parent path even
    # when output_dir doesn't yet exist so the parents resolve.
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(go_vendor_dir, output_dir, dirs_exist_ok=True)

    # Prune first-party subtrees. The first-party prefix is a directory
    # path under output_dir — e.g. github.com/ai-dynamo/.
    pruned = 0
    for prefix in _FIRST_PARTY_GO_PREFIXES:
        fp_root = output_dir / prefix.rstrip("/")
        if fp_root.is_dir():
            shutil.rmtree(fp_root)
            pruned += 1

    # Approximate count of remaining modules: every leaf directory
    # under output_dir is one module's source tree. Cheap to compute,
    # useful for logs.
    module_dirs = [
        d
        for d in output_dir.rglob("*")
        if d.is_dir() and any(c.suffix == ".go" for c in d.iterdir() if c.is_file())
    ]
    logger.info(
        "Go sources collected: ~%d module dirs (pruned %d first-party prefixes from %s)",
        len(module_dirs),
        pruned,
        go_vendor_dir,
    )
    return {
        "declared": len(module_dirs),
        "collected": len(module_dirs),
        "missing": [],
        "skipped": [],
        "first_party_excluded": pruned,
    }


# First-party native components — NVIDIA-authored from-source builds.
# Source lives in an ai-dynamo repo; we don't ship it in OSRB archives
# because we're the upstream author, not redistributing someone else's
# OSS. Mirrors _FIRST_PARTY_RUST_PREFIXES, which already treats nixl-* as
# first-party. Match against the leading filename token (everything before
# the first hyphen + version).
_FIRST_PARTY_NATIVE_NAMES = {"cuda-checkpoint-helper", "nixl"}


def collect_native_sources(workspace_native_dir: Path, output_dir: Path) -> dict:
    """Copy third-party native source archives preserved by builder stages.

    Each builder Dockerfile that does `RUN git clone …` or `wget …tar` should
    preserve the resulting archive at /tmp/native-sources/<name>-<version>.tar.gz
    (or a per-component subdirectory). The Dockerfile's `sources_collect`
    stage `COPY --from=<builder> /tmp/native-sources/ /opt/native-sources/`
    puts them where this script can find them.

    First-party components (cuda-checkpoint-helper, nixl) are filtered out:
    NVIDIA-authored, source on GitHub, not OSS redistribution.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if not workspace_native_dir.is_dir():
        logger.warning(
            "No native source directory at %s; skipping.", workspace_native_dir
        )
        return {"declared": 0, "collected": 0, "missing": [], "skipped": []}
    copied = 0
    skipped_first_party = 0
    for item in workspace_native_dir.iterdir():
        # Each entry is either a per-component subdir (snapshot's
        # criu/, cuda-checkpoint/ pattern) or a flat tarball
        # (<name>-<version>.tar.gz). A first-party match is the
        # whole name OR a <name>-<version>{.tar.gz,/} prefix.
        name_for_filter = item.name
        if any(
            name_for_filter == fp or name_for_filter.startswith(fp + "-")
            for fp in _FIRST_PARTY_NATIVE_NAMES
        ):
            skipped_first_party += 1
            continue
        dest = output_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
        copied += 1
    logger.info(
        "Native sources collected: %d entries (skipped %d first-party) from %s",
        copied,
        skipped_first_party,
        workspace_native_dir,
    )
    if copied == 0:
        logger.warning(
            "native ecosystem is enabled but %s yielded no entries; no native "
            "source is in the archive",
            workspace_native_dir,
        )
    return {
        "declared": copied,
        "collected": copied,
        "missing": [],
        "skipped": [],
        "first_party_excluded": skipped_first_party,
    }


_README_TEMPLATE = """\
# Source archive — OSRB submission companion

Generated by container/compliance/collect_sources.py during the
nightly / RC / release container build.

## Contents

Each top-level directory corresponds to one ecosystem's third-party
sources that the runtime image redistributes. Directories appear only
when the corresponding ecosystem was collected (depends on which
runtime this archive belongs to).

| Directory | What's here |
|---|---|
| `dpkg/`    | `.dsc` + tarballs for Debian/Ubuntu source packages behind the binary packages we install on top of the baseline image. Scoped to the delta against the baseline SBOM, then collapsed onto source packages. Vendor-proprietary packages (CUDA / DOCA / Mellanox repos) publish no source and are not included; see `manifest.json`. |
| `rust/`    | `cargo vendor` tree filtered to the third-party crates that appear in the installed wheels' embedded SBOMs. Excludes first-party crates (`dynamo-*`, `kvbm-*`, `nixl-*`) — those are NVIDIA-authored and source is public at github.com/ai-dynamo. Includes the workspace `Cargo.toml` + `Cargo.lock` for context. |
| `go/`      | `go mod vendor` tree for the operator / snapshot / EPP binaries. Excludes first-party modules (`github.com/ai-dynamo/...`). |
| `native/`  | Upstream source tarballs (or git clones) for from-source builds — CRIU, cuda-checkpoint, UCX, libfabric, gdrcopy, nv-codec-headers, FFmpeg where applicable. Excludes first-party native components (`cuda-checkpoint-helper`, `nixl`). |

## Coverage

`manifest.json` records, per ecosystem, how many components were
declared, how many were collected, and the name and reason for every
one that was not. Collection is best-effort and never fails the build,
so this file — not the presence of a directory — is the authoritative
statement of what this archive does and does not cover.

## Python sources are not in this archive

Python wheels are source-distributable by definition (a `.whl` is a
zip of `.py` source files, per PEP 427). Every Python package we ship
is installed into `${VIRTUAL_ENV}/lib/python*/site-packages/` inside
the runtime image, and the source files live there directly. The
runtime image already redistributes Python source by construction — a
separate `python/` directory in this archive would be duplication.

Auditor note: for Python packages with compiled C-extensions, the
compiled `.so` is in the wheel but the `.c` source is not. The common
case for our deps is pure-Python or PyTorch-built C-extensions where
PyTorch ships the source separately. If OSRB needs C-extension source
for a specific package, fetch it from PyPI's sdist (e.g.
`pip download --no-binary :all: <package>`).

## Verifying this archive

`CHECKSUMS.sha256` lists every file shipped here, so after extracting:

    sha256sum -c CHECKSUMS.sha256

The companion OSRB bundle's `build-provenance.json` records the
SHA-256 of the `sources.tar.gz` this tree was extracted from, under
the `sources.sha256` field, so tampering between bundle and sources is
detectable:

    sha256sum sources.tar.gz
"""


def write_readme(sources_root: Path) -> None:
    """Write a small README explaining the archive structure."""
    (sources_root / "README.md").write_text(_README_TEMPLATE, encoding="utf-8")


def write_manifest(sources_root: Path, stats: dict[str, dict]) -> None:
    """Record per-ecosystem coverage so gaps are visible rather than silent."""
    total_declared = sum(s.get("declared", 0) for s in stats.values())
    total_collected = sum(s.get("collected", 0) for s in stats.values())
    doc = {
        "ecosystems": stats,
        "totals": {
            "declared": total_declared,
            "collected": total_collected,
            "missing": sum(len(s.get("missing", [])) for s in stats.values()),
            "skipped": sum(len(s.get("skipped", [])) for s in stats.values()),
        },
    }
    (sources_root / "manifest.json").write_text(
        json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for eco, s in sorted(stats.items()):
        declared, collected = s.get("declared", 0), s.get("collected", 0)
        if declared and collected < declared:
            logger.warning(
                "%s coverage: %d/%d (%.1f%%) — see manifest.json",
                eco,
                collected,
                declared,
                100.0 * collected / declared,
            )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(sources_root: Path) -> int:
    """sha256 every file in the tree, in sorted order, to CHECKSUMS.sha256.

    Sorted relative paths make the listing reproducible across builds, which
    is what lets the OSRB bundle record a single digest over the whole tree.
    """
    lines: list[str] = []
    for path in sorted(sources_root.rglob("*")):
        if not path.is_file() or path.name == "CHECKSUMS.sha256":
            continue
        rel = path.relative_to(sources_root).as_posix()
        lines.append(f"{_file_sha256(path)}  {rel}")
    (sources_root / "CHECKSUMS.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return len(lines)


def pack_sources_tar(sources_root: Path, tar_path: Path) -> None:
    """Pack sources_root as a deterministic gzip tarball.

    Members are added in sorted order with pinned mtimes and neutral
    ownership, and gzip's own mtime header is zeroed, so two builds of the
    same sources produce the same bytes. That is what makes recording a
    single sha256 in the OSRB bundle meaningful.
    """
    if not sources_root.is_dir():
        raise FileNotFoundError(f"sources root missing: {sources_root}")

    def _normalise(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mtime = 0
        return info

    paths = sorted(p for p in sources_root.rglob("*") if p.is_file())
    # gzip's mtime header has to be pinned separately from the tar member
    # mtimes, so drive GzipFile directly instead of tarfile's "w:gz" mode.
    with (
        tar_path.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6, mtime=0) as gz,
        tarfile.open(fileobj=gz, mode="w", format=tarfile.GNU_FORMAT) as tar,
    ):
        for path in paths:
            arcname = path.relative_to(sources_root.parent).as_posix()
            tar.add(path, arcname=arcname, filter=_normalise)
    logger.info(
        "Wrote %s (%.1f MB, %d files)",
        tar_path,
        tar_path.stat().st_size / 1e6,
        len(paths),
    )


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
        "--output-tar",
        type=Path,
        default=Path("/sources.tar.gz"),
        help="Where to write the final sources tarball.",
    )
    parser.add_argument(
        "--sources-root",
        type=Path,
        default=Path("/sources"),
        help="Working directory for collected sources, packed into --output-tar.",
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
        "--dpkg-root",
        type=Path,
        default=Path("/"),
        help=(
            "Filesystem root whose /var/lib/dpkg database defines the dpkg "
            "packages to archive source for. Defaults to the current container "
            "root. Use this when the sources stage installs helper packages "
            "that are not shipped in the final image."
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
            "Globbed for lib/python*/site-packages/*.dist-info/sboms/*.cyclonedx.json "
            "to discover the third-party crate set we ship. Mutually "
            "exclusive with --rust-site-packages."
        ),
    )
    parser.add_argument(
        "--rust-site-packages",
        type=Path,
        default=None,
        help=(
            "Direct path to a site-packages (or dist-packages) directory. "
            "Use this when packages are installed system-wide via "
            "`pip install --break-system-packages` (sglang's pattern) "
            "rather than into a venv with the standard lib/python*/site-packages "
            "layout. Overrides --rust-venv when set."
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
    parser.add_argument(
        "--go-vendor-dir",
        type=Path,
        default=Path("/opt/go-vendor"),
        help=(
            "Go vendor tree produced by `go mod vendor` in the Go builder "
            "stage. First-party modules (github.com/ai-dynamo/...) are "
            "pruned from the output; everything else is copied as-is."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s [%(name)s]: %(message)s",
    )

    ecosystems = args.ecosystem or ["dpkg", "native"]

    # Reject typo'd ecosystem names rather than silently collecting nothing for
    # them and shipping an incomplete source archive (e.g. "rdust").
    valid_ecosystems = {"dpkg", "rust", "go", "native"}
    unknown = [e for e in ecosystems if e not in valid_ecosystems]
    if unknown:
        parser.error(
            f"unknown --ecosystem value(s): {', '.join(sorted(unknown))}; "
            f"valid: {', '.join(sorted(valid_ecosystems))}"
        )

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

    stats: dict[str, dict] = {}
    if "dpkg" in ecosystems:
        stats["dpkg"] = collect_dpkg_sources(
            base_sbom, args.sources_root / "dpkg", dpkg_root=args.dpkg_root
        )
    if "rust" in ecosystems:
        if args.rust_site_packages is not None:
            site_dirs = [args.rust_site_packages]
        else:
            site_dirs = list(args.rust_venv.glob("lib/python*/site-packages"))
        stats["rust"] = collect_rust_sources(
            site_dirs, args.rust_vendor_full, args.sources_root / "rust"
        )
    if "go" in ecosystems:
        stats["go"] = collect_go_sources(args.go_vendor_dir, args.sources_root / "go")
    if "native" in ecosystems:
        stats["native"] = collect_native_sources(
            args.native_source_dir, args.sources_root / "native"
        )

    write_readme(args.sources_root)
    write_manifest(args.sources_root, stats)
    file_count = write_checksums(args.sources_root)
    pack_sources_tar(args.sources_root, args.output_tar)
    logger.info(
        "Source archival complete: %d files; %s",
        file_count,
        {
            eco: f"{s.get('collected', 0)}/{s.get('declared', 0)}"
            for eco, s in stats.items()
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
