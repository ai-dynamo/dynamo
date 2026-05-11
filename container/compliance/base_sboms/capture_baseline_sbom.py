#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture a baseline SBOM and verify a from-image is built on top of it.

Used at base-adoption time and when refreshing a tracked baseline. The
captured SBOM is the **baseline's** full slim CycloneDX, not a delta.
Subtraction happens at NOTICES-render time inside each runtime's
licenses stage; this tool just ensures the baseline file exists and
the manifest entry records the relationship.

Compliance model (one level deep):
  - from_image     The image our Dockerfile FROMs (e.g. vllm/vllm-openai).
  - baseline_image The floor of OUR compliance responsibility, specified
                   by the engineer adopting the base (e.g. cuda-dl-base).
                   Everything below this line is the upstream owner's
                   responsibility; everything above is ours to attribute.

Usage:

  capture_baseline_sbom.py \\
    --from vllm/vllm-openai:v0.12.0 \\
    --baseline nvcr.io/nvidia/cuda-dl-base:25.06-cuda12.9-devel-ubuntu24.04

Steps performed:
  1. Resolve both image:tag refs to manifest-list digests
     (sha256 of `imagetools inspect --raw` output, per OCI spec).
  2. Fetch per-platform layer digest lists for both images.
  3. **Layer-prefix verification**: assert from_image's layers start with
     baseline_image's full layer list. If not, fail loudly — the engineer
     said "this is built on X" but the bytes say otherwise.
  4. syft scan the baseline only. Apply slim filter
     (drop properties/hashes/dependencies; keep evidence). Hard cap 4 MB.
     Write to base_sboms/<short>@<digest8>.cdx.json if not already
     present at the recorded digest.
  5. syft scan the from_image (in memory; not persisted). Compute
     delta components = from.components - baseline.components by
     (name, version). This is what we'd redistribute on top of the
     baseline.
  6. Run policy/validate.py against the delta. Any denied or UNKNOWN
     license fails the capture: engineer must add an override / exception,
     or pick a different from_image.
  7. Add / update the manifest.json entry recording both digests and
     the baseline_sbom filename.

Flags:
  --dry-run                 Do all checks, write nothing.
  --platform linux/amd64    Pin platform for layer-prefix + syft.
                            Default linux/amd64; multi-arch entries get
                            separate manifest rows when needed.
  --skip-layer-prefix-check Escape hatch for vendors who squash layers.
                            Records the override in the manifest entry
                            for auditability. Use only with justification.

Dependencies on the runner:
  - docker buildx (registry manifest resolution)
  - syft (`syft scan -o cyclonedx-json --platform <p> <ref>`)
  - python3 with the compliance package importable

Exit codes:
  0  capture (or dry-run) completed cleanly
  1  layer-prefix or policy validation failed
  2  registry / syft / I/O failure
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make container/compliance importable when running this script directly
# from the repo without `pip install -e`.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from container.compliance.policy import validate as policy_validate  # noqa: E402

logger = logging.getLogger(__name__)

_CORPUS_DIR = Path(__file__).resolve().parent
_MANIFEST_PATH = _CORPUS_DIR / "manifest.json"
_POLICY_PATH = _CORPUS_DIR.parent / "policy" / "licenses.toml"
_SIZE_CAP_BYTES = 4 * 1024 * 1024  # 4 MB; 1 MB headroom under GitLab's 5 MB pain point


# ---- registry: digest + layer resolution -------------------------------------


def _imagetools_raw(ref: str) -> bytes:
    """Return the canonical manifest bytes for `ref` from the registry."""
    result = subprocess.run(
        ["docker", "buildx", "imagetools", "inspect", "--raw", ref],
        check=True,
        capture_output=True,
    )
    return result.stdout


def resolve_index_digest(ref: str) -> str:
    """SHA-256 of the canonical manifest bytes = the OCI manifest digest."""
    raw = _imagetools_raw(ref)
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def resolve_platform_layers(ref: str, platform: str) -> list[str]:
    """Return the ordered layer digest list for `ref` on `platform`.

    Handles both manifest-list (multi-arch) and single-manifest images.
    For multi-arch, picks the platform entry and fetches its manifest
    separately.
    """
    raw = _imagetools_raw(ref)
    parsed = json.loads(raw)

    if "manifests" in parsed:
        os_part, arch_part = platform.split("/", 1)
        for entry in parsed["manifests"]:
            p = entry.get("platform", {}) or {}
            if p.get("os") == os_part and p.get("architecture") == arch_part:
                # Fetch this platform's specific manifest.
                repo = ref.rsplit(":", 1)[0]
                platform_ref = f"{repo}@{entry['digest']}"
                manifest_raw = _imagetools_raw(platform_ref)
                manifest = json.loads(manifest_raw)
                return [layer["digest"] for layer in manifest.get("layers", [])]
        raise ValueError(
            f"no platform {platform} in {ref}; available: "
            + ", ".join(
                f"{e.get('platform', {}).get('os')}/{e.get('platform', {}).get('architecture')}"
                for e in parsed["manifests"]
            )
        )

    # Single-platform manifest
    return [layer["digest"] for layer in parsed.get("layers", [])]


# ---- syft scan + slim filter -------------------------------------------------


def syft_scan(ref: str, platform: str) -> dict:
    """Run syft against image:tag for a specific platform, return parsed CycloneDX."""
    env = {**os.environ, "SYFT_PLATFORM": platform}
    result = subprocess.run(
        ["syft", "scan", "-o", "cyclonedx-json", ref],
        check=True,
        capture_output=True,
        env=env,
    )
    return json.loads(result.stdout)


def slim_cyclonedx(doc: dict) -> dict:
    """Drop properties/hashes from components; drop the dependencies graph.

    Keeps `components[].evidence` -- the paths where each package was
    found inside the image are critical for audit.
    """
    out = dict(doc)
    out.pop("dependencies", None)
    components = out.get("components", []) or []
    slimmed = []
    for c in components:
        c_out = {k: v for k, v in c.items() if k not in ("properties", "hashes")}
        slimmed.append(c_out)
    out["components"] = slimmed
    return out


# ---- delta computation + policy validation -----------------------------------


def _component_key(c: dict) -> tuple[str, str]:
    return (c.get("name") or "", c.get("version") or "")


import re as _re

# Canonicalization map for the prose-form license names syft emits in the
# CycloneDX `{"license": {"name": "..."}}` shape. Without this, every
# variant of "Apache 2.0 License" / "BSD License" / "MIT-License" turns
# into a different LicenseRef-* string and the policy gate explodes.
#
# Keys are normalized to: lowercased, hyphens-to-spaces, whitespace
# collapsed. Both "NVIDIA-Proprietary-Software" and "nvidia proprietary
# software" normalize to "nvidia proprietary software".
_CANONICAL_NAME_MAP: dict[str, str] = {
    # Apache family
    "apache": "Apache-2.0",
    "apache 2": "Apache-2.0",
    "apache 2.0": "Apache-2.0",
    "apache software": "Apache-2.0",
    "apache 2.0 with llvm exceptions": "Apache-2.0 WITH LLVM-exception",
    "apache 2 with llvm exceptions": "Apache-2.0 WITH LLVM-exception",
    "apache 2 llvm exceptions": "Apache-2.0 WITH LLVM-exception",
    # BSD family
    "bsd": "BSD-3-Clause",
    "bsd 2 clause": "BSD-2-Clause",
    "bsd 3 clause": "BSD-3-Clause",
    "3 clause bsd": "BSD-3-Clause",
    "modified bsd": "BSD-3-Clause",
    "new bsd": "BSD-3-Clause",
    "revised bsd": "BSD-3-Clause",
    "bsd2": "BSD-2-Clause",
    "bsd3": "BSD-3-Clause",
    # MIT / Expat
    "mit": "MIT",
    "expat": "MIT",  # Expat IS MIT (FSF naming)
    # ISC
    "isc": "ISC",
    # LGPL family — denied by policy, but canonicalize so the policy
    # message says "denied: LGPL-3.0-or-later" rather than "unknown ref".
    "lgpl": "LGPL-3.0-or-later",
    "lgpl v3": "LGPL-3.0-or-later",
    "lgplv3": "LGPL-3.0-or-later",
    "lgpl 3": "LGPL-3.0-or-later",
    "lgpl 2.1": "LGPL-2.1-or-later",
    "gnu lgpl": "LGPL-3.0-or-later",
    # GPL family — same logic
    "gpl": "GPL-3.0-or-later",
    "gnu gpl": "GPL-3.0-or-later",
    "gpl v3": "GPL-3.0-or-later",
    "gpl v2": "GPL-2.0-or-later",
    "gplv3": "GPL-3.0-or-later",
    "gplv2": "GPL-2.0-or-later",
    # More Apache prose variants
    "apache license": "Apache-2.0",
    "apache license 2.0": "Apache-2.0",
    "apache license version 2.0": "Apache-2.0",
    "apache license, version 2.0": "Apache-2.0",
    # PSF
    "python": "PSF-2.0",
    "python software foundation": "PSF-2.0",
    "psf": "PSF-2.0",
    # CC0 / Public domain
    "cc0": "CC0-1.0",
    "cc0 1.0": "CC0-1.0",
    "public domain": "LicenseRef-public-domain",
    # NVIDIA variants
    "nvidia proprietary": "LicenseRef-NVIDIA-Proprietary",
    "nvidia proprietary software": "LicenseRef-NVIDIA-Proprietary",
    # Misc
    "zlib": "Zlib",
    "boost": "BSL-1.0",
    "boost software": "BSL-1.0",
    "mpl 2.0": "MPL-2.0",
    "mozilla public 2.0": "MPL-2.0",
    "unicode": "Unicode-3.0",
    "unicode 3.0": "Unicode-3.0",
    "artistic": "Artistic-2.0",
}

# Tokens syft emits from copyright-file paragraph headers that aren't
# actually license names — they're prose words ("Permission is hereby
# granted...", "Redistribution and use...", "By submitting...") that
# happen to follow a "License:" pseudo-header. Dropping them silently
# matches what a careful auditor would do: these aren't licenses.
_NON_LICENSE_TOKENS: frozenset[str] = frozenset({
    "by", "permission", "redistribution", "this", "the", "see", "for",
    "unknown", "various", "free", "other", "dual",
})

# Some syft outputs use sha256:HEX content fingerprints as license "names"
# when it couldn't identify the license but wants a stable tag. These are
# not licenses at all — drop them.
_CONTENT_HASH_RE = _re.compile(r"^sha\d+:[0-9a-f]{16,}$", _re.IGNORECASE)


def _canonicalize_license_name(name: str) -> str | None:
    """Map a syft `license.name` string to canonical SPDX, or None to drop.

    Returns:
      - A canonical SPDX expression for recognized prose ("MIT License").
      - None for tokens that aren't license names at all (parser artifacts,
        content-hash fingerprints).
      - A LicenseRef-<sanitized> identifier for genuinely unknown licenses
        we should surface for human review rather than silently drop.
    """
    s = name.strip()
    if not s:
        return None

    # Drop sha256:HEX-style content-hash entries syft emits when it can't
    # name the license. Also drop bare "sha256" / "sha1" / etc.
    if _CONTENT_HASH_RE.match(s) or s.lower().startswith(("sha256", "sha1:", "md5:")):
        return None

    # Strip a trailing " License" / "-License" suffix so "Apache 2.0 License"
    # and "Apache 2.0" both hit the same map entry.
    base = s
    for suffix in (" License", "-License", " license", "-license"):
        if base.endswith(suffix):
            base = base[: -len(suffix)].rstrip(" -")
            break

    # Normalize for key lookup: lowercase, hyphens-to-spaces, collapse
    # whitespace, drop punctuation that breaks SPDX parsing.
    key = base.lower().replace("-", " ")
    for ch in ",()/":
        key = key.replace(ch, " ")
    key = " ".join(key.split())
    if key in _NON_LICENSE_TOKENS:
        return None
    if key in _CANONICAL_NAME_MAP:
        return _CANONICAL_NAME_MAP[key]

    # Expat-<anything> → MIT. Expat IS MIT (FSF naming); the suffixes
    # we see in the wild ("Expat-Intel", "Expat-NVIDIA", "Expat-RedHat",
    # "Expat-(MIT/X11)") are vendor copyright tags on permissive MIT
    # text. All map to plain MIT for policy purposes.
    if key.startswith("expat ") or key == "expat":
        return "MIT"

    # Last resort: surface as LicenseRef-<sanitized name>. Sanitization
    # strips characters that break SPDX parsers (colons, commas, parens,
    # slashes) and collapses runs of hyphens. Use the post-suffix-strip
    # form so "Apache-2.0 License" and "Apache-2.0" converge.
    sanitized = base.replace(" ", "-")
    for ch in ":,()/":
        sanitized = sanitized.replace(ch, "-")
    sanitized = _re.sub(r"-+", "-", sanitized).strip("-")
    if not sanitized:
        return None
    return f"LicenseRef-{sanitized}"


def _extract_spdx(component: dict) -> str:
    """Render the component's license[] array to a single SPDX-ish expression."""
    raw = component.get("licenses") or []
    if not raw:
        evidence = component.get("evidence") or {}
        raw = evidence.get("licenses") or []
    if not raw:
        return "UNKNOWN"

    parts: list[str] = []
    for entry in raw:
        if "expression" in entry:
            parts.append(entry["expression"])
        elif "license" in entry:
            inner = entry["license"]
            if "id" in inner:
                parts.append(inner["id"])
            elif "name" in inner:
                canonical = _canonicalize_license_name(inner["name"])
                if canonical is not None:
                    parts.append(canonical)
    if not parts:
        return "UNKNOWN"
    if len(parts) == 1:
        return parts[0]
    return " AND ".join(f"({p})" if " " in p else p for p in parts)


def _ecosystem_from_purl(purl: str) -> str:
    """Map a purl prefix to one of our policy ecosystems."""
    if not purl:
        return "unknown"
    if purl.startswith("pkg:deb/"):
        return "dpkg"
    if purl.startswith("pkg:pypi/"):
        return "python"
    if purl.startswith("pkg:cargo/"):
        return "rust"
    if purl.startswith("pkg:golang/"):
        return "go"
    if purl.startswith("pkg:rpm/"):
        return "rpm"
    return "unknown"


def compute_delta(from_sbom: dict, baseline_sbom: dict) -> list[dict]:
    """Components in from_sbom but not baseline_sbom, keyed by (name, version)."""
    baseline_keys = {_component_key(c) for c in baseline_sbom.get("components", [])}
    return [
        c
        for c in from_sbom.get("components", [])
        if _component_key(c) not in baseline_keys
    ]


def validate_delta(delta: list[dict], policy_path: Path) -> tuple[int, list[str]]:
    """Run policy/validate.validate_row on every delta component.

    Returns (violation_count, formatted_violation_lines).
    """
    policy = policy_validate.load_policy(policy_path)
    violations: list[str] = []
    for component in delta:
        name = component.get("name") or ""
        version = str(component.get("version") or "")
        purl = component.get("purl") or ""
        ecosystem = _ecosystem_from_purl(purl)
        spdx = _extract_spdx(component)
        if ecosystem == "unknown":
            # syft emits operating-system components and other non-package
            # entries -- skip those (no ecosystem the policy gates).
            continue
        v = policy_validate.validate_row(policy, ecosystem, name, version, spdx)
        if v is not None:
            violations.append(str(v))
    return len(violations), violations


# ---- manifest I/O ------------------------------------------------------------


def _short_name(image: str) -> str:
    """nvcr.io/nvidia/cuda-dl-base -> cuda-dl-base"""
    return image.rsplit("/", 1)[-1]


def load_manifest(path: Path) -> dict:
    if not path.is_file():
        return {"schema_version": 1, "entries": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(manifest: dict, path: Path) -> None:
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")


def upsert_entry(manifest: dict, entry: dict) -> None:
    """Replace any existing entry matching (from_image, from_tag, platform), else append."""
    key = (entry["from_image"], entry["from_tag"], entry["platform"])
    entries = manifest.setdefault("entries", [])
    for i, existing in enumerate(entries):
        if (
            existing.get("from_image"),
            existing.get("from_tag"),
            existing.get("platform"),
        ) == key:
            entries[i] = entry
            return
    entries.append(entry)


# ---- main capture orchestration ----------------------------------------------


def split_ref(ref: str) -> tuple[str, str]:
    if ":" not in ref:
        raise ValueError(f"expected image:tag, got {ref!r}")
    image, tag = ref.rsplit(":", 1)
    return image, tag


def capture(
    from_ref: str,
    baseline_ref: str,
    platform: str,
    dry_run: bool,
    skip_layer_prefix_check: bool,
    corpus_dir: Path,
    manifest_path: Path,
    policy_path: Path,
) -> int:
    from_image, from_tag = split_ref(from_ref)
    baseline_image, baseline_tag = split_ref(baseline_ref)

    logger.info("Resolving registry digests...")
    try:
        from_digest = resolve_index_digest(from_ref)
        baseline_digest = resolve_index_digest(baseline_ref)
    except subprocess.CalledProcessError as exc:
        logger.error("Registry inspect failed: %s", (exc.stderr or b"").decode())
        return 2
    logger.info("  from_image     %s @ %s", from_ref, from_digest)
    logger.info("  baseline_image %s @ %s", baseline_ref, baseline_digest)

    if not skip_layer_prefix_check:
        logger.info("Verifying layer-prefix relationship (platform=%s)...", platform)
        try:
            baseline_layers = resolve_platform_layers(baseline_ref, platform)
            from_layers = resolve_platform_layers(from_ref, platform)
        except (subprocess.CalledProcessError, ValueError) as exc:
            logger.error("Layer resolution failed: %s", exc)
            return 2
        n = len(baseline_layers)
        if from_layers[:n] != baseline_layers:
            logger.error(
                "Layer-prefix mismatch: %s is not built on %s.\n"
                "  baseline has %d layers; from-image's first %d layers do not match.\n"
                "  baseline layers: %s\n"
                "  from layers[:%d]: %s",
                from_ref,
                baseline_ref,
                n,
                n,
                baseline_layers,
                n,
                from_layers[:n],
            )
            return 1
        logger.info(
            "  OK: %d baseline layers are a prefix of %d from-image layers",
            n,
            len(from_layers),
        )

    logger.info("Running syft on baseline...")
    try:
        baseline_sbom = syft_scan(baseline_ref, platform)
    except subprocess.CalledProcessError as exc:
        logger.error("syft scan failed on baseline: %s", (exc.stderr or b"").decode())
        return 2

    logger.info("Running syft on from-image (for delta validation; not persisted)...")
    try:
        from_sbom = syft_scan(from_ref, platform)
    except subprocess.CalledProcessError as exc:
        logger.error("syft scan failed on from-image: %s", (exc.stderr or b"").decode())
        return 2

    delta = compute_delta(from_sbom, baseline_sbom)
    logger.info(
        "Delta: %d components in %s not in %s",
        len(delta),
        from_ref,
        baseline_ref,
    )

    logger.info("Validating delta against %s...", policy_path)
    violation_count, violation_lines = validate_delta(delta, policy_path)
    if violation_count:
        logger.error(
            "Delta validation FAILED (%d violation%s). "
            "Either reject this from-image, or add overrides/exceptions:",
            violation_count,
            "" if violation_count == 1 else "s",
        )
        for line in violation_lines:
            logger.error("  %s", line)
        return 1
    logger.info("  OK: all %d delta components pass policy", len(delta))

    # Apply slim filter, compute filename, write (unless --dry-run).
    slim = slim_cyclonedx(baseline_sbom)
    short = _short_name(baseline_image)
    digest8 = baseline_digest.removeprefix("sha256:")[:8]
    sbom_filename = f"{short}@{digest8}.cdx.json"
    sbom_path = corpus_dir / sbom_filename

    payload = json.dumps(slim, separators=(",", ":"))
    payload_bytes = payload.encode("utf-8")
    size = len(payload_bytes)
    if size > _SIZE_CAP_BYTES:
        logger.error(
            "Slim SBOM exceeds 4 MB cap (%.2f MB). "
            "Per-ecosystem split is supported by the schema but not yet "
            "implemented in this tool -- file a follow-up.",
            size / (1024 * 1024),
        )
        return 1
    logger.info(
        "Slim baseline SBOM: %s (%.1f KB, %d components)",
        sbom_filename,
        size / 1024,
        len(slim.get("components", [])),
    )

    entry = {
        "from_image": from_image,
        "from_tag": from_tag,
        "from_digest": from_digest,
        "baseline_image": baseline_image,
        "baseline_tag": baseline_tag,
        "baseline_digest": baseline_digest,
        "baseline_sbom": sbom_filename,
        "platform": platform,
    }
    if skip_layer_prefix_check:
        entry["layer_prefix_check_skipped"] = True

    if dry_run:
        logger.info("--dry-run: not writing %s or manifest", sbom_path)
        logger.info("manifest entry would be:\n%s", json.dumps(entry, indent=2))
        return 0

    sbom_path.write_bytes(payload_bytes)
    manifest = load_manifest(manifest_path)
    upsert_entry(manifest, entry)
    save_manifest(manifest, manifest_path)
    logger.info("Wrote %s and updated %s", sbom_path, manifest_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="capture_baseline_sbom",
        description="Capture a baseline SBOM and verify a from-image is built on it.",
    )
    parser.add_argument(
        "--from",
        dest="from_ref",
        required=True,
        help="from_image:tag (the FROM line in our Dockerfile)",
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="baseline_image:tag (engineer-specified compliance floor)",
    )
    parser.add_argument(
        "--platform",
        default="linux/amd64",
        help="Platform for layer-prefix verification + syft (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all checks but write nothing. Use for vetting candidates.",
    )
    parser.add_argument(
        "--skip-layer-prefix-check",
        action="store_true",
        help="Disable the layer-prefix invariant check. Use only when the vendor squashes layers.",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=_CORPUS_DIR,
        help="Where SBOM files are written (default: %(default)s)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_MANIFEST_PATH,
        help="Manifest file to update (default: %(default)s)",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=_POLICY_PATH,
        help="Policy file to validate the delta against (default: %(default)s)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        return capture(
            from_ref=args.from_ref,
            baseline_ref=args.baseline,
            platform=args.platform,
            dry_run=args.dry_run,
            skip_layer_prefix_check=args.skip_layer_prefix_check,
            corpus_dir=args.corpus_dir,
            manifest_path=args.manifest,
            policy_path=args.policy,
        )
    except Exception:  # pragma: no cover
        logger.exception("Unhandled error during capture")
        return 2


if __name__ == "__main__":
    sys.exit(main())
