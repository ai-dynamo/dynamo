# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Media-codec allowlist gate for finished images.

We ship an in-tree ffmpeg built to a narrow media-codec set, so the shipped
filesystem should not contain media-codec libraries/binaries from anywhere else.
Because SBOMs miss statically-bundled codec `.so` files, this is primarily a
FILESYSTEM scan of the built image; an optional `--sbom` adds a CycloneDX
component/version gate (an ffmpeg version floor).

A denylist hit is classified as:
  - ALLOWED    — under an `allow_paths` prefix (our own in-tree ffmpeg)
  - EXCEPTION  — matches a reasoned `exceptions` entry (logged, does not fail)
  - VIOLATION  — anything else

With --fail-on-findings, any VIOLATION exits non-zero, failing the image build
when this runs in the compliance licenses stage.

Policy: container/compliance/policy/codec_policy.yaml
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import os
import sys
from pathlib import Path

import yaml

try:
    from packaging.version import Version
except ImportError:  # packaging is optional; fall back to a numeric-tuple compare
    Version = None

logger = logging.getLogger("compliance.scan_codecs")

# Virtual kernel filesystems only — pruned at the scan root because walking them
# is meaningless (and /proc is effectively unbounded). Real directories such as
# /tmp and /run are NOT pruned: a codec binary left there still ships in the
# image and must be scanned.
_PRUNE_DIRS = {"proc", "sys", "dev"}


def _glob_to_fullpath_pattern(glob: str) -> str:
    """Translate a `**/`-style policy glob into an fnmatch pattern applied to the
    full POSIX path. `**/` becomes `*` (fnmatch's `*` already spans `/`), so
    `**/libx264.so*` -> `*libx264.so*` and `**/nvidia/dali/.libs/libav*` ->
    `*nvidia/dali/.libs/libav*`."""
    return "*" + glob[3:] if glob.startswith("**/") else glob


def _matches_any(path: str, globs: list[str]) -> str | None:
    for g in globs:
        if fnmatch.fnmatch(path, _glob_to_fullpath_pattern(g)):
            return g
    return None


class CodecPolicy:
    def __init__(self, doc: dict):
        self.deny_globs: list[str] = doc.get("deny_globs", []) or []
        self.allow_paths: list[str] = doc.get("allow_paths", []) or []
        self.deny_components: list[dict] = doc.get("deny_components", []) or []
        self.exceptions: list[dict] = doc.get("exceptions", []) or []

    @classmethod
    def load(cls, path: Path) -> "CodecPolicy":
        # Fail closed: a compliance gate that silently passes every image because
        # of a typo'd/missing key is worse than no gate. Validate the shape up
        # front so a malformed policy aborts the scan instead of allow-listing.
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            raise ValueError(f"codec policy {path} must be a YAML mapping")
        if not isinstance(doc.get("deny_globs"), list) or not doc["deny_globs"]:
            raise ValueError(f"codec policy {path} requires a non-empty deny_globs list")
        for exc in doc.get("exceptions") or []:
            if not all(exc.get(k) for k in ("glob", "reason", "owner")):
                raise ValueError(
                    f"codec policy {path}: each exception needs glob, reason, and owner"
                )
        return cls(doc)

    def classify(self, abspath: str) -> tuple[str, str | None]:
        """Return (verdict, detail) for a path already known to hit a deny glob.
        verdict is 'allowed' | 'exception' | 'violation'."""
        if any(abspath.startswith(p) for p in self.allow_paths):
            return "allowed", None
        for exc in self.exceptions:
            if fnmatch.fnmatch(abspath, _glob_to_fullpath_pattern(exc.get("glob", ""))):
                return "exception", (exc.get("reason") or "").strip()
        return "violation", None


def scan_filesystem(root: Path, policy: CodecPolicy):
    """Walk `root`, returning (violations, exceptions, allowed) lists of dicts."""
    violations, exceptions, allowed = [], [], []
    root_str = os.path.normpath(str(root))
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # Prune virtual dirs only at the scan root (don't prune a legitimately
        # named 'run'/'tmp' deeper in a package tree).
        if os.path.normpath(dirpath) == root_str:
            dirnames[:] = [d for d in dirnames if d not in _PRUNE_DIRS]
        for name in filenames:
            full = os.path.join(dirpath, name)
            # Report paths as absolute-from-root ("/usr/...") so allow_paths and
            # exceptions read naturally regardless of where root is mounted.
            rel = os.path.relpath(full, root_str)
            abspath = "/" + rel.replace(os.sep, "/")
            hit = _matches_any(abspath, policy.deny_globs)
            if not hit:
                continue
            verdict, detail = policy.classify(abspath)
            entry = {"path": abspath, "glob": hit, "detail": detail}
            {"allowed": allowed, "exception": exceptions, "violation": violations}[
                verdict
            ].append(entry)
    return violations, exceptions, allowed


def _version_lt(a: str, b: str) -> bool:
    """a < b. Uses packaging.version when available, else a numeric-tuple fallback.
    A malformed version under packaging raises (InvalidVersion) and fails the scan
    rather than silently degrading to the naive comparator."""
    if Version is not None:
        return Version(a) < Version(b)

    def key(v: str) -> list[int]:
        return [int(x) if x.isdigit() else 0 for x in v.replace("-", ".").split(".")]

    return key(a) < key(b)


def scan_sbom(sbom_path: Path, policy: CodecPolicy) -> list[dict]:
    """Flag SBOM components matching deny_components (optionally below a fixed
    version). Returns a list of violation dicts."""
    doc = json.loads(sbom_path.read_text(encoding="utf-8"))
    out: list[dict] = []
    denied = {d["name"].lower(): d for d in policy.deny_components if d.get("name")}
    for comp in doc.get("components", []) or []:
        name = (comp.get("name") or "").lower()
        rule = denied.get(name)
        if not rule:
            continue
        version = str(comp.get("version") or "")
        floor = rule.get("min_fixed_version")
        if floor:
            # No version ⇒ we cannot prove the component is at/above the fixed
            # floor, so flag it rather than let an unversioned denied component
            # (e.g. an ffmpeg with no version in the SBOM) silently pass.
            if not version:
                out.append(
                    {
                        "path": f"sbom:{name}@(no version)",
                        "glob": f"version unknown, cannot verify >= {floor}",
                        "detail": f"no version in SBOM; fixed floor is {floor}",
                    }
                )
            elif _version_lt(version, floor):
                out.append(
                    {
                        "path": f"sbom:{name}@{version}",
                        "glob": f"version < {floor}",
                        "detail": f"below fixed version {floor}",
                    }
                )
        else:
            out.append(
                {
                    "path": f"sbom:{name}@{version}",
                    "glob": "denied component",
                    "detail": None,
                }
            )
    return out


def _emit_summary(image, violations, exceptions, allowed):
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary:
        return
    with open(summary, "a", encoding="utf-8") as s:
        s.write(f"### Codec gate{' — ' + image if image else ''}\n\n")
        s.write(
            f"- violations: **{len(violations)}** | "
            f"exceptions: {len(exceptions)} | allowed (in-tree): {len(allowed)}\n\n"
        )
        if violations:
            s.write("| path | matched | note |\n|---|---|---|\n")
            for v in violations:
                s.write(f"| `{v['path']}` | `{v['glob']}` | {v['detail'] or ''} |\n")
        else:
            s.write("✅ no media-codec artifacts outside the allowlist.\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="compliance.scan_codecs")
    p.add_argument(
        "--root", type=Path, default=Path("/"), help="filesystem root to scan"
    )
    p.add_argument(
        "--policy",
        type=Path,
        default=Path(__file__).parent / "policy" / "codec_policy.yaml",
    )
    p.add_argument(
        "--sbom",
        type=Path,
        default=None,
        help="optional CycloneDX SBOM for the component gate",
    )
    p.add_argument("--image", default="", help="image label for the report header")
    p.add_argument("--report", type=Path, default=None)
    p.add_argument(
        "--fail-on-findings", action="store_true", help="exit non-zero on any violation"
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s [%(name)s]: %(message)s",
    )

    policy = CodecPolicy.load(args.policy)
    violations, exceptions, allowed = scan_filesystem(args.root, policy)
    if args.sbom:
        # An explicitly-supplied SBOM that is missing must fail loudly — silently
        # skipping it would disable the version-floor gate without anyone noticing.
        if not args.sbom.is_file():
            logger.error("--sbom given but not found: %s", args.sbom)
            return 1
        violations.extend(scan_sbom(args.sbom, policy))

    hdr = f"Codec gate{' for ' + args.image if args.image else ''}"
    print(hdr)
    print(
        f"  scanned root: {args.root} | violations: {len(violations)} | "
        f"exceptions: {len(exceptions)} | allowed (in-tree): {len(allowed)}"
    )
    for label, rows in (
        ("ALLOWED", allowed),
        ("EXCEPTION", exceptions),
        ("VIOLATION", violations),
    ):
        for r in rows:
            line = f"  {label:9} {r['path']}"
            if r.get("detail"):
                line += f"  — {r['detail']}"
            (logger.debug if label == "ALLOWED" else print)(line)

    if args.report:
        args.report.write_text(
            "\n".join(f"{r['path']}\t{r['glob']}" for r in violations) + "\n",
            encoding="utf-8",
        )
    _emit_summary(args.image, violations, exceptions, allowed)

    if violations:
        logger.error(
            "%d media-codec artifact(s) present outside the allowlist; "
            "remove them or add a reasoned exception to codec_policy.yaml",
            len(violations),
        )
        if args.fail_on_findings:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
