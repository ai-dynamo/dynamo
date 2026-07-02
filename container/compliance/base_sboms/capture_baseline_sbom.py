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
      (drop properties/hashes/dependencies; keep evidence). Hard cap 5 MB.
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
import re as _re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Make container/compliance importable when running this script directly
# from the repo without `pip install -e`.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from container.compliance import overrides as license_overrides  # noqa: E402
from container.compliance.policy import validate as policy_validate  # noqa: E402

logger = logging.getLogger(__name__)

_CORPUS_DIR = Path(__file__).resolve().parent
_MANIFEST_PATH = _CORPUS_DIR / "manifest.json"
_POLICY_PATH = _CORPUS_DIR.parent / "policy" / "licenses.toml"
_SIZE_CAP_BYTES = (
    5 * 1024 * 1024
)  # 5 MB cap for large XPU baselines; keep aligned with CI artifact constraints
_DEFAULT_FROM_SBOM_CACHE_DIR = (
    Path(os.environ.get("TMPDIR", "/tmp")) / "dynamo-compliance-syft-cache"
)

# Fixed namespace for deterministic baseline-BOM serial numbers. Any constant
# UUID works; this one is dedicated to dynamo compliance baselines so the
# derived UUIDv5 serials are stable and never collide with other uuid5 callers.
_SERIAL_NAMESPACE = uuid.UUID("d9b2d63d-a233-4123-847e-0a3fbf6b1a77")


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


def _deterministic_serial_number(doc: dict) -> str | None:
    """Derive a stable ``urn:uuid:`` serial from the scanned image identity.

    syft stamps every scan with a random ``serialNumber`` and a wall-clock
    ``metadata.timestamp``; both churn on every run and would swamp the diff
    when a baseline is refreshed. We replace the serial with a UUIDv5 over the
    image's ``metadata.component`` name:version so re-capturing an unchanged
    image is byte-identical, matching the deterministic-serial pattern used in
    ``osrb/package.py``. Returns None when no component identity is present, in
    which case the caller drops the field entirely.
    """
    component = (doc.get("metadata") or {}).get("component") or {}
    name = component.get("name")
    if not name:
        return None
    version = component.get("version")
    seed = f"{name}:{version}" if version else name
    return f"urn:uuid:{uuid.uuid5(_SERIAL_NAMESPACE, seed)}"


def slim_cyclonedx(doc: dict) -> dict:
    """Drop properties/hashes from components; drop the dependencies graph; and
    normalize the two non-deterministic document fields.

    ``serialNumber`` is replaced with a deterministic UUIDv5 derived from the
    scanned image identity (or dropped when no identity is present), and the
    wall-clock ``metadata.timestamp`` is dropped. Both otherwise change on every
    syft run, so normalizing them lets an unchanged image re-capture to a zero
    diff -- the whole point of committing these baselines.

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
    # Own the component order rather than inheriting syft's. syft's intrinsic
    # ordering can shift between releases, which would churn the entire file
    # even when the image is unchanged; sorting here keeps re-capture diffs
    # proportional to actual package changes (and survives syft upgrades).
    # bom-ref is a deterministic tiebreaker for duplicate (name, version) rows.
    slimmed.sort(
        key=lambda c: (
            str(c.get("name", "")),
            str(c.get("version", "")),
            str(c.get("type", "")),
            str(c.get("purl", "")),
            str(c.get("bom-ref", "")),
        )
    )
    out["components"] = slimmed

    serial = _deterministic_serial_number(out)
    if serial is not None:
        out["serialNumber"] = serial
    else:
        out.pop("serialNumber", None)
    metadata = out.get("metadata")
    if isinstance(metadata, dict):
        metadata = dict(metadata)
        metadata.pop("timestamp", None)
        out["metadata"] = metadata
    return out


# ---- delta computation + policy validation -----------------------------------


def _component_key(c: dict) -> tuple[str, str, str]:
    # Include ecosystem (from the purl type): a pypi and a cargo package can
    # share a name+version, and keying on (name, version) alone would wrongly
    # subtract one ecosystem's component against the other's in the delta.
    return (
        _ecosystem_from_purl(c.get("purl") or ""),
        c.get("name") or "",
        c.get("version") or "",
    )


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
_NON_LICENSE_TOKENS: frozenset[str] = frozenset(
    {
        "by",
        "permission",
        "redistribution",
        "this",
        "the",
        "see",
        "for",
        "unknown",
        "various",
        "free",
        "other",
        "dual",
    }
)

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

    # Last resort: surface as LicenseRef-<sanitized name>. SPDX grammar
    # restricts LicenseRef ID body to [0-9A-Za-z.-], so we strip every
    # other character that syft DEP-5 output drops in there (commas,
    # parens, slashes, colons, tildes, plus signs — all surfaced from
    # Ubuntu's copyright files). Anything not in the allowed set
    # becomes a hyphen; runs of hyphens collapse to one.
    sanitized = _re.sub(r"[^0-9A-Za-z.-]", "-", base)
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
    """Components in from_sbom but not baseline_sbom, keyed by (ecosystem, name, version)."""
    baseline_keys = {_component_key(c) for c in baseline_sbom.get("components", [])}
    return [
        c
        for c in from_sbom.get("components", [])
        if _component_key(c) not in baseline_keys
    ]


def validate_delta(delta: list[dict], policy_path: Path) -> tuple[int, list[str]]:
    """Run policy/validate.validate_row on every delta component.

    For each component, the SPDX expression is resolved as:
      1. license_overrides.yaml (by ecosystem, name) — authoritative when
         we've explicitly verified a package's license. The same map our
         runtime generators (dpkg.py, python.py) consult; reusing it here
         means a single source of truth for hand-curated license facts.
      2. syft's CycloneDX output, normalized via _extract_spdx +
         _canonicalize_license_name.

    Returns (violation_count, formatted_violation_lines).
    """
    policy = policy_validate.load_policy(policy_path)
    violations: list[str] = []
    for component in delta:
        name = component.get("name") or ""
        version = str(component.get("version") or "")
        purl = component.get("purl") or ""
        ecosystem = _ecosystem_from_purl(purl)
        if ecosystem == "unknown":
            # syft emits operating-system components and other non-package
            # entries -- skip those (no ecosystem the policy gates).
            continue
        # Hand-curated override beats syft's heuristic.
        override = license_overrides.lookup(ecosystem, name, version)
        spdx = override if override else _extract_spdx(component)
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
    reuse_cached_sbom: bool = False,
    from_sbom_cache_dir: Path | None = None,
    no_from_sbom_cache: bool = False,
) -> int:
    from_image, from_tag = split_ref(from_ref)
    baseline_image, baseline_tag = split_ref(baseline_ref)

    if from_sbom_cache_dir is None:
        from_sbom_cache_dir = _DEFAULT_FROM_SBOM_CACHE_DIR

    logger.info("Resolving registry digests...")
    try:
        from_digest = resolve_index_digest(from_ref)
        baseline_digest = resolve_index_digest(baseline_ref)
    except subprocess.CalledProcessError as exc:
        logger.error("Registry inspect failed: %s", (exc.stderr or b"").decode())
        return 2
    logger.info("  from_image     %s @ %s", from_ref, from_digest)
    logger.info("  baseline_image %s @ %s", baseline_ref, baseline_digest)

    # Cache-key paths. Both keys include the manifest digest, so a rebuilt
    # vendor tag (different digest) naturally invalidates the cache — no
    # explicit invalidation is needed.
    short = _short_name(baseline_image)
    baseline_digest_hex = baseline_digest.removeprefix("sha256:")
    # Arch suffix: baseline_digest is the multi-arch INDEX digest (same for
    # every platform), so the per-platform SBOMs must be disambiguated by arch
    # or they'd collide. The runtime licenses stage selects the matching file
    # via ${BASELINE_SBOM_STEM}-${TARGETARCH}.cdx.json.
    arch = platform.rsplit("/", 1)[-1]
    # baseline_stem is what context.yaml's `baseline_sbom:` holds; the licenses
    # stage / audit append -${TARGETARCH}.cdx.json. Surface it explicitly below
    # so a re-capture doesn't leave the engineer to hand-strip the arch suffix.
    baseline_stem = f"{short}@{baseline_digest_hex[:8]}"
    sbom_filename = f"{baseline_stem}-{arch}.cdx.json"
    sbom_path = corpus_dir / sbom_filename
    from_digest_hex = from_digest.removeprefix("sha256:")
    from_cache_filename = f"{from_digest_hex}-{platform.replace('/', '_')}.cdx.json"
    from_cache_path = from_sbom_cache_dir / from_cache_filename

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

    # --- Baseline SBOM: corpus file is the cache. The filename embeds the
    # baseline's manifest digest, and OCI digests are immutable — same digest
    # means same bytes — so reusing an existing file is safe by construction.
    if reuse_cached_sbom and sbom_path.is_file():
        baseline_sbom = json.loads(sbom_path.read_text(encoding="utf-8"))
        if not baseline_sbom.get("components"):
            logger.warning(
                "Cached baseline SBOM at %s has no components[]; ignoring cache and running syft.",
                sbom_path,
            )
            reuse_cached_sbom = False
        else:
            logger.info(
                "Baseline SBOM cache hit: %s (skipping syft scan, %d components)",
                sbom_filename,
                len(baseline_sbom.get("components", [])),
            )
    if not (reuse_cached_sbom and sbom_path.is_file()):
        logger.info("Running syft on baseline...")
        try:
            baseline_sbom = syft_scan(baseline_ref, platform)
        except subprocess.CalledProcessError as exc:
            logger.error(
                "syft scan failed on baseline: %s", (exc.stderr or b"").decode()
            )
            return 2

    # --- From-image SBOM: persisted cache under a TMPDIR by default. Not
    # committed to the repo (different per-runner) but survives across
    # back-to-back invocations of this script, which is the slow path we
    # care about when iterating on overrides/exceptions.
    from_sbom = None
    if not no_from_sbom_cache and from_cache_path.is_file():
        try:
            from_sbom = json.loads(from_cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "From-image SBOM cache at %s unreadable (%s); falling through to syft.",
                from_cache_path,
                exc,
            )
            from_sbom = None
        if from_sbom is not None and not from_sbom.get("components"):
            logger.warning(
                "Cached from-image SBOM has no components[]; ignoring cache and running syft."
            )
            from_sbom = None
        if from_sbom is not None:
            logger.info(
                "From-image SBOM cache hit: %s (%d components)",
                from_cache_path,
                len(from_sbom.get("components", [])),
            )
    if from_sbom is None:
        logger.info(
            "Running syft on from-image (for delta validation; not persisted)..."
        )
        try:
            from_sbom = syft_scan(from_ref, platform)
        except subprocess.CalledProcessError as exc:
            logger.error(
                "syft scan failed on from-image: %s", (exc.stderr or b"").decode()
            )
            return 2
        if not no_from_sbom_cache:
            try:
                from_sbom_cache_dir.mkdir(parents=True, exist_ok=True)
                from_cache_path.write_text(
                    json.dumps(from_sbom, separators=(",", ":")),
                    encoding="utf-8",
                )
                logger.info("Cached from-image SBOM to %s", from_cache_path)
            except OSError as exc:
                logger.warning(
                    "Could not write from-image SBOM cache to %s (%s); continuing without cache.",
                    from_cache_path,
                    exc,
                )

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

    # Apply slim filter; the filename + path were computed up front so the
    # baseline-cache check could short-circuit syft.
    slim = slim_cyclonedx(baseline_sbom)

    # Pretty-print with sorted object keys (matching manifest.json) so the
    # committed baselines produce readable, line-oriented diffs when a base
    # is refreshed. slim_cyclonedx also sorts the component array so the order
    # is ours (stable across syft upgrades). The slim filter keeps these
    # comfortably under the size cap even indented.
    payload = json.dumps(slim, indent=2, sort_keys=True) + "\n"
    payload_bytes = payload.encode("utf-8")
    size = len(payload_bytes)
    if size > _SIZE_CAP_BYTES:
        logger.error(
            "Slim SBOM exceeds 5 MB cap (%.2f MB). "
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
    logger.info(
        "Set this framework's baseline_sbom in container/context.yaml to: %s",
        baseline_stem,
    )
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
    parser.add_argument(
        "--reuse-cached-sbom",
        action="store_true",
        help=(
            "If the baseline SBOM file already exists in the corpus dir for "
            "this baseline's manifest digest, load it instead of re-running "
            "syft. Safe because OCI digests are immutable — same digest means "
            "same bytes."
        ),
    )
    parser.add_argument(
        "--from-sbom-cache-dir",
        type=Path,
        default=_DEFAULT_FROM_SBOM_CACHE_DIR,
        help=(
            "Directory for cached from-image SBOMs, keyed by manifest digest "
            "+ platform. Default: %(default)s. Cache hits skip the slow syft "
            "scan on the from-image (5-10 min on large framework images)."
        ),
    )
    parser.add_argument(
        "--no-from-sbom-cache",
        action="store_true",
        help="Disable the from-image SBOM cache entirely (do not read or write).",
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
            reuse_cached_sbom=args.reuse_cached_sbom,
            from_sbom_cache_dir=args.from_sbom_cache_dir,
            no_from_sbom_cache=args.no_from_sbom_cache,
        )
    except Exception:  # pragma: no cover
        logger.exception("Unhandled error during capture")
        return 2


if __name__ == "__main__":
    sys.exit(main())
