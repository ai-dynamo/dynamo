# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Augment a syft-generated CycloneDX BOM with source-compiled binaries.

syft reliably catalogues distro and language-ecosystem packages, but binaries
that Dynamo compiles from upstream source during the container build (NATS,
etcd, nixl, UCX, gdrcopy, libfabric, FFmpeg, aws-efa-installer, flashinfer,
lmcache, ...) are invisible to it. This script reads a CycloneDX BOM,
``container/context.yaml`` (the single source of truth for those versions),
and ``binary_refs.yaml`` (the registry of upstream repos), then injects new
CycloneDX components of type ``library`` with PURLs of the form
``pkg:github/<org>/<repo>@<version>``. License text is fetched from the
upstream GitHub repository via the REST API (authenticated when
``GITHUB_TOKEN`` is set, anonymous otherwise).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import yaml

GITHUB_API = "https://api.github.com"
USER_AGENT = "dynamo-sbom-inject/1.0"

# VCS base URLs by PURL type
_VCS_BASE: dict[str, str] = {
    "github": "https://github.com",
    "gitlab": "https://gitlab.com",
    "bitbucket": "https://bitbucket.org",
}


def _gh_request(path: str) -> dict[str, Any] | None:
    """GET a GitHub API path. Returns parsed JSON or None on any failure."""
    url = f"{GITHUB_API}{path}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        # OSError covers ConnectionResetError, socket.timeout, etc.
        print(f"  [warn] GitHub API {path} failed: {e}", file=sys.stderr)
        return None


def fetch_license_text(repo: str) -> tuple[str | None, str | None]:
    """Return (spdx_id, license_text) for ``<org>/<repo>`` or (None, None)."""
    data = _gh_request(f"/repos/{repo}/license")
    if not data:
        return (None, None)
    spdx = (data.get("license") or {}).get("spdx_id")
    content_b64 = data.get("content") or ""
    encoding = data.get("encoding") or ""
    text: str | None = None
    if encoding == "base64" and content_b64:
        try:
            text = base64.b64decode(content_b64).decode("utf-8", errors="replace")
        except (ValueError, UnicodeDecodeError):
            text = None
    return (spdx if spdx and spdx != "NOASSERTION" else None, text)


def load_context_versions(context_path: Path, framework: str) -> dict[str, str]:
    """Return a flat dict of all scalar values from context.yaml for ``framework``.

    Extracts every str|int|float value in the framework section, not just
    keys ending with _version/_ref, to support varied naming conventions.
    """
    with open(context_path, "r", encoding="utf-8") as f:
        ctx = yaml.safe_load(f)
    section = ctx.get(framework) or {}
    versions: dict[str, str] = {}
    for key, val in section.items():
        if isinstance(val, (str, int, float)):
            versions[key] = str(val)
    return versions


def build_component(key: str, entry: dict[str, Any], version: str) -> dict[str, Any]:
    """Assemble one CycloneDX component from a binary_refs entry."""
    repo = entry["repo"]
    name = entry["name"]
    purl_type = entry.get("purl_type", "github")
    purl = f"pkg:{purl_type}/{repo}@{version}"

    # Resolve VCS base URL from purl_type
    vcs_base = _VCS_BASE.get(purl_type)
    if vcs_base is None:
        raise ValueError(f"unsupported purl_type: {purl_type}")

    spdx_id, text = fetch_license_text(repo)
    if not spdx_id:
        spdx_id = entry.get("license", "NOASSERTION")

    license_entry: dict[str, Any] = {"license": {"id": spdx_id}}
    if text:
        license_entry["license"]["text"] = {
            "contentType": "text/plain",
            "content": text,
        }

    return {
        "bom-ref": f"source-binary:{name}@{version}",
        "type": "library",
        "name": name,
        "version": version,
        "purl": purl,
        "licenses": [license_entry],
        "externalReferences": [{"type": "vcs", "url": f"{vcs_base}/{repo}"}],
        "properties": [
            {"name": "dynamo:source", "value": "context.yaml"},
            {"name": "dynamo:context_key", "value": key},
        ],
    }


def inject(
    bom: dict[str, Any],
    context_path: Path,
    binary_refs_path: Path,
    framework: str,
) -> int:
    """Mutate ``bom`` in place, return the number of components added.

    Raises RuntimeError if a binary_refs entry whose applies_to includes
    the current framework cannot resolve a version from context.yaml.
    """
    with open(binary_refs_path, "r", encoding="utf-8") as f:
        registry = yaml.safe_load(f) or {}

    versions = load_context_versions(context_path, framework)
    components = bom.setdefault("components", [])
    existing_purls = {c.get("purl") for c in components if c.get("purl")}

    added = 0
    for key, entry in registry.items():
        applies_to = entry.get("applies_to") or []
        if framework not in applies_to:
            continue
        version = versions.get(key)
        if not version:
            raise RuntimeError(
                f"binary_refs[{key}] applies to {framework} but "
                f"context.yaml[{framework}] has no entry for '{key}'"
            )
        component = build_component(key, entry, version)
        if component["purl"] in existing_purls:
            print(f"  [skip] {component['purl']}: already in BOM", file=sys.stderr)
            continue
        components.append(component)
        existing_purls.add(component["purl"])
        added += 1
        print(f"  [add]  {component['purl']}", file=sys.stderr)
    return added


def _deterministic_serial(framework: str, commit_sha: str, bom: dict[str, Any]) -> str:
    """Generate deterministic UUID5 serial number from framework, commit, and PURLs."""
    components = bom.get("components") or []
    purls = sorted(c.get("purl", "") for c in components if c.get("purl"))
    seed = f"dynamo-sbom:{framework}:{commit_sha}:{','.join(purls)}"
    return f"urn:uuid:{uuid5(NAMESPACE_URL, seed)}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bom", required=True, type=Path, help="Input/output CycloneDX JSON"
    )
    parser.add_argument(
        "--context",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "context.yaml",
        help="Path to container/context.yaml",
    )
    parser.add_argument(
        "--binary-refs",
        type=Path,
        default=Path(__file__).with_name("binary_refs.yaml"),
        help="Path to binary_refs.yaml",
    )
    parser.add_argument(
        "--framework",
        required=True,
        choices=["dynamo", "vllm", "sglang", "trtllm"],
        help="Context framework being augmented",
    )
    parser.add_argument(
        "--commit-sha",
        default=os.environ.get("DYNAMO_COMMIT_SHA", "HEAD"),
        help="Commit SHA for deterministic serial number (default: env DYNAMO_COMMIT_SHA or HEAD)",
    )
    args = parser.parse_args()

    with open(args.bom, "r", encoding="utf-8") as f:
        bom = json.load(f)

    added = inject(bom, args.context, args.binary_refs, args.framework)

    # Generate deterministic serial number after injection
    if "serialNumber" not in bom:
        bom["serialNumber"] = _deterministic_serial(
            args.framework, args.commit_sha, bom
        )

    with open(args.bom, "w", encoding="utf-8") as f:
        json.dump(bom, f, indent=2)
        f.write("\n")
    print(f"injected {added} source-binary component(s) -> {args.bom}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
