# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""License resolution for Cargo, Go, and PyPI lockfile entries.

Cargo.lock and go.mod do not carry license metadata, and syft sometimes
fails to find a license for a Python package even when one is published.
This module fills both gaps by querying the upstream registries:

* Cargo crates: `https://crates.io/api/v1/crates/<name>/<version>` returns
  the SPDX license expression in ``version.license``.
* Go modules: `https://api.deps.dev/v3alpha/systems/go/packages/<encoded module>/versions/<version>`
  returns ``licenses[]`` (SPDX identifiers). deps.dev is the public Google-run
  mirror; proxy.golang.org itself does not expose license metadata.
* PyPI distributions: `https://pypi.org/pypi/<name>/<version>/json` returns
  ``info.license_expression`` (SPDX, newest), ``info.license`` (free text,
  sometimes SPDX), and ``info.classifiers`` (Trove identifiers we map to
  SPDX as a last resort).

Lookups are cached in a SQLite file (default
``~/.cache/dynamo-compliance/license-lookup.sqlite``). The cache is per-machine
and intentionally not committed - it can always be rebuilt from the registries.

This module is import-safe even when the network is unreachable: the resolver
catches all transport errors and returns ``UNKNOWN``, so callers can always
fall back to the original behavior.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from contextlib import closing
from typing import Iterable

DEFAULT_CACHE_PATH = os.path.expanduser(
    "~/.cache/dynamo-compliance/license-lookup.sqlite"
)
USER_AGENT = (
    "dynamo-compliance-license-lookup/1.0 (+https://github.com/ai-dynamo/dynamo)"
)
HTTP_TIMEOUT = 10  # seconds per request

# crates.io asks for <= 1 req/sec without an API token; we throttle conservatively.
_CRATES_IO_THROTTLE_SECS = 1.0
# deps.dev has no published rate limit; throttle lightly to be polite.
_DEPS_DEV_THROTTLE_SECS = 0.05
# PyPI's JSON API has no documented per-second limit; throttle lightly.
_PYPI_THROTTLE_SECS = 0.05

# Trove classifier -> SPDX identifier. PyPI ``info.classifiers`` is the most
# reliable license source since it comes from a fixed vocabulary; the freer
# ``info.license`` field is often a paragraph of prose. Only the families we
# actually see in the Dynamo image set are mapped; unknown entries fall back
# to UNKNOWN rather than guessing.
_TROVE_TO_SPDX = {
    "License :: OSI Approved :: Apache Software License": "Apache-2.0",
    "License :: OSI Approved :: Apache 2.0 License": "Apache-2.0",
    "License :: OSI Approved :: MIT License": "MIT",
    "License :: OSI Approved :: MIT No Attribution License (MIT-0)": "MIT-0",
    "License :: OSI Approved :: BSD License": "BSD-3-Clause",
    "License :: OSI Approved :: ISC License (ISCL)": "ISC",
    "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "License :: OSI Approved :: Python Software Foundation License": "PSF-2.0",
    "License :: OSI Approved :: GNU General Public License v2 (GPLv2)": "GPL-2.0-only",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)": "GPL-3.0-only",
    "License :: OSI Approved :: GNU Lesser General Public License v2 (LGPLv2)": "LGPL-2.0-only",
    "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)": "LGPL-3.0-only",
    "License :: OSI Approved :: GNU Affero General Public License v3": "AGPL-3.0-only",
    "License :: OSI Approved :: Zlib/libpng License": "Zlib",
    "License :: Public Domain": "Unlicense",
    "License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication": "CC0-1.0",
}
# Emit a progress line every N successful network fetches.
_PROGRESS_EVERY = 50

_last_call: dict[str, float] = {}


def _throttle(host: str, min_gap: float) -> None:
    now = time.monotonic()
    prev = _last_call.get(host, 0.0)
    delay = min_gap - (now - prev)
    if delay > 0:
        time.sleep(delay)
    _last_call[host] = time.monotonic()


def _open_cache(path: str | None = None) -> sqlite3.Connection:
    path = path or DEFAULT_CACHE_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS license_lookup (
            ecosystem TEXT NOT NULL,
            name      TEXT NOT NULL,
            version   TEXT NOT NULL,
            license   TEXT NOT NULL,
            fetched_at INTEGER NOT NULL,
            PRIMARY KEY (ecosystem, name, version)
        )
        """
    )
    conn.commit()
    return conn


def _cache_get(
    conn: sqlite3.Connection, ecosystem: str, name: str, version: str
) -> str | None:
    """Return a cached license, treating UNKNOWN as a miss.

    UNKNOWN is a "we couldn't resolve it last time" placeholder, not a stable
    answer. Treating it as a miss means each new run picks up improvements to
    the resolver (e.g. a newly added GitHub fallback) without needing the
    user to manually invalidate the cache.
    """
    cur = conn.execute(
        "SELECT license FROM license_lookup WHERE ecosystem=? AND name=? AND version=?",
        (ecosystem, name, version),
    )
    row = cur.fetchone()
    if not row or row[0] == "UNKNOWN":
        return None
    return row[0]


def _cache_put(
    conn: sqlite3.Connection,
    ecosystem: str,
    name: str,
    version: str,
    license_id: str,
) -> None:
    """Persist a resolved license. UNKNOWN values are intentionally skipped so
    a future run with a smarter resolver gets a clean retry."""
    if license_id == "UNKNOWN":
        return
    conn.execute(
        """
        INSERT OR REPLACE INTO license_lookup
            (ecosystem, name, version, license, fetched_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (ecosystem, name, version, license_id, int(time.time())),
    )
    conn.commit()


def _http_json(url: str, *, headers: dict[str, str] | None = None) -> dict | None:
    # OSError covers urllib.error.URLError/HTTPError and the builtin TimeoutError;
    # JSONDecodeError covers a 200 with a malformed body.
    request_headers = {"User-Agent": USER_AGENT}
    if headers:
        request_headers.update(headers)
    req = urllib.request.Request(url, headers=request_headers)
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def fetch_cargo_license(name: str, version: str) -> str:
    """Look up an SPDX license for a crate version on crates.io.

    Returns ``"UNKNOWN"`` for any failure (network down, package missing,
    license field empty). The crates.io API includes the license in the
    response body even for yanked versions.
    """
    if not name or not version:
        return "UNKNOWN"
    _throttle("crates.io", _CRATES_IO_THROTTLE_SECS)
    safe_name = urllib.parse.quote(name, safe="")
    safe_version = urllib.parse.quote(version, safe="")
    url = f"https://crates.io/api/v1/crates/{safe_name}/{safe_version}"
    payload = _http_json(url)
    if not payload:
        return "UNKNOWN"
    version_obj = payload.get("version") or {}
    lic = version_obj.get("license")
    return lic.strip() if isinstance(lic, str) and lic.strip() else "UNKNOWN"


def fetch_go_license(module: str, version: str) -> str:
    """Look up an SPDX license for a Go module version via deps.dev.

    Returns ``"UNKNOWN"`` for any failure. deps.dev returns one or more SPDX
    identifiers in ``licenses[]``; we join multiples with ``" AND "`` to match
    the SPDX expression style used elsewhere in the pipeline.
    """
    if not module or not version:
        return "UNKNOWN"
    _throttle("deps.dev", _DEPS_DEV_THROTTLE_SECS)
    safe_module = urllib.parse.quote(module, safe="")
    safe_version = urllib.parse.quote(version, safe="")
    url = (
        "https://api.deps.dev/v3alpha/systems/go/packages/"
        f"{safe_module}/versions/{safe_version}"
    )
    payload = _http_json(url)
    if not payload:
        return "UNKNOWN"
    licenses = payload.get("licenses") or []
    cleaned = [s.strip() for s in licenses if isinstance(s, str) and s.strip()]
    if not cleaned:
        return "UNKNOWN"
    return " AND ".join(cleaned)


def _strip_pep440_local(version: str) -> str:
    """Drop the PEP 440 local-version segment (``+something``).

    PyPI never publishes builds whose version contains a local segment, so
    looking up ``flash-attn==2.7.4.post1+nv26.2.44259020`` always 404s even
    though the upstream release ``2.7.4.post1`` is on PyPI with full license
    metadata. Stripping after the ``+`` lets the upstream lookup succeed.
    """
    base, _, _local = version.partition("+")
    return base


def fetch_pypi_license(name: str, version: str) -> str:
    """Look up an SPDX license for a PyPI distribution.

    Tries three fields in order, preferring the most reliable:
      1. ``info.license_expression`` - new in PEP 639, always SPDX.
      2. ``info.classifiers`` - Trove identifiers from a fixed vocabulary,
         mapped via ``_TROVE_TO_SPDX``.
      3. ``info.license`` - free-form prose; only used if it looks short
         and SPDX-shaped (no spaces, all printable ASCII), since maintainers
         frequently dump the entire LICENSE file in here.

    If the exact version is not on PyPI (404, common for ``+cu129``-style
    local builds republished by NVIDIA), a single retry is made against the
    PEP 440 base version before giving up.

    Returns ``"UNKNOWN"`` for any failure or unrecognized value.
    """
    if not name or not version:
        return "UNKNOWN"
    _throttle("pypi.org", _PYPI_THROTTLE_SECS)
    safe_name = urllib.parse.quote(name, safe="")
    safe_version = urllib.parse.quote(version, safe="")
    url = f"https://pypi.org/pypi/{safe_name}/{safe_version}/json"
    payload = _http_json(url)
    if not payload:
        base_version = _strip_pep440_local(version)
        if base_version != version:
            _throttle("pypi.org", _PYPI_THROTTLE_SECS)
            safe_base = urllib.parse.quote(base_version, safe="")
            payload = _http_json(
                f"https://pypi.org/pypi/{safe_name}/{safe_base}/json"
            )
        if not payload:
            return "UNKNOWN"
    info = payload.get("info") or {}
    expr = info.get("license_expression")
    if isinstance(expr, str) and expr.strip():
        return expr.strip()
    classifiers = info.get("classifiers") or []
    spdx_from_classifiers = [
        _TROVE_TO_SPDX[c] for c in classifiers if c in _TROVE_TO_SPDX
    ]
    if spdx_from_classifiers:
        unique = sorted(set(spdx_from_classifiers))
        return " AND ".join(unique) if len(unique) > 1 else unique[0]
    raw = info.get("license")
    if isinstance(raw, str):
        candidate = raw.strip()
        if (
            candidate
            and " " not in candidate
            and candidate.isascii()
            and len(candidate) < 64
        ):
            return candidate
    # Final fallback: ask GitHub for the repo's licensee-detected SPDX. PyPI's
    # JSON often omits PEP 639 fields even when the wheel METADATA has them,
    # but most maintainers link a github.com URL in project_urls.
    return _github_license_for_pypi(info)


_GITHUB_REPO_RE = re.compile(r"https?://github\.com/([^/?#]+)/([^/?#]+)")


def _github_license_for_pypi(info: dict) -> str:
    """Resolve a PyPI distribution's license via the GitHub repo it links to.

    Walks ``info.home_page`` and every value in ``info.project_urls`` looking
    for a ``github.com/<owner>/<repo>`` link. The first one that matches is
    queried via ``GET /repos/{owner}/{repo}/license``. Returns ``"UNKNOWN"``
    when no GitHub URL is present, the API fails, or the repo's license is
    ``NOASSERTION`` (GitHub's value for custom/unrecognized LICENSE files).
    """
    candidates: list[str] = []
    home = info.get("home_page")
    if isinstance(home, str):
        candidates.append(home)
    for value in (info.get("project_urls") or {}).values():
        if isinstance(value, str):
            candidates.append(value)
    for candidate in candidates:
        match = _GITHUB_REPO_RE.match(candidate)
        if not match:
            continue
        owner = match.group(1)
        repo = match.group(2).removesuffix(".git")
        spdx = _github_repo_license(owner, repo)
        if spdx and spdx != "NOASSERTION":
            return spdx
        # Stop after the first GitHub link; subsequent ones rarely differ.
        return "UNKNOWN"
    return "UNKNOWN"


def _github_repo_license(owner: str, repo: str) -> str | None:
    """Fetch the SPDX id GitHub's licensee assigns to a repo's LICENSE file."""
    _throttle("api.github.com", _PYPI_THROTTLE_SECS)
    safe_owner = urllib.parse.quote(owner, safe="")
    safe_repo = urllib.parse.quote(repo, safe="")
    url = f"https://api.github.com/repos/{safe_owner}/{safe_repo}/license"
    payload = _http_json(url, headers={"Accept": "application/vnd.github+json"})
    if not payload:
        return None
    spdx = (payload.get("license") or {}).get("spdx_id")
    return spdx if isinstance(spdx, str) and spdx else None


_FETCHERS = {
    "cargo": fetch_cargo_license,
    "golang": fetch_go_license,
    "pypi": fetch_pypi_license,
}


def resolve_licenses(
    entries: Iterable[dict],
    *,
    cache_path: str | None = None,
) -> list[dict]:
    """Annotate lockfile entries with a ``license`` field.

    ``entries`` is an iterable of dicts with at least ``name``, ``version``,
    and ``ecosystem`` keys (the shape produced by ``cargo.parse_cargo_lock``
    and ``gomod.parse_go_mod``). Each returned dict is a shallow copy with
    ``license`` set; existing ``license`` values are preserved.

    Lookups are cached in SQLite at ``cache_path`` (defaulting to
    ``DEFAULT_CACHE_PATH``); cached entries are returned instantly and unknown
    ones trigger a network fetch. Anything we cannot resolve becomes
    ``"UNKNOWN"``.
    """
    materialized = list(entries)
    if not materialized:
        return materialized

    out: list[dict] = []
    misses = 0
    with closing(_open_cache(cache_path)) as conn:
        for i, entry in enumerate(materialized, 1):
            new = dict(entry)
            existing = new.get("license")
            if existing and existing != "UNKNOWN":
                out.append(new)
                continue
            ecosystem = entry.get("ecosystem", "")
            name = entry.get("name", "")
            version = entry.get("version", "")
            cached = _cache_get(conn, ecosystem, name, version)
            if cached is not None:
                new["license"] = cached
                out.append(new)
                continue
            fetcher = _FETCHERS.get(ecosystem)
            if fetcher is None:
                new["license"] = "UNKNOWN"
                out.append(new)
                continue
            license_id = fetcher(name, version)
            _cache_put(conn, ecosystem, name, version, license_id)
            new["license"] = license_id
            misses += 1
            if misses % _PROGRESS_EVERY == 0:
                print(
                    f"  license_lookup: fetched {misses} new entries "
                    f"({i}/{len(materialized)} processed)",
                    file=sys.stderr,
                )
            out.append(new)
    if misses:
        print(
            f"  license_lookup: {misses} fetched, "
            f"{len(materialized) - misses} from cache",
            file=sys.stderr,
        )
    return out


def dedupe_packages(packages: list[dict]) -> list[dict]:
    """Collapse duplicate (name, version) entries, preferring non-UNKNOWN license.

    Syft sometimes reports the same Python package twice when it is installed
    in two paths (e.g. system site-packages plus a venv). The post-processor
    surfaces both rows. This helper keeps a stable single row per (name,
    version), preferring the variant with a resolved license over UNKNOWN.

    Order is preserved by first occurrence.
    """
    seen: dict[tuple[str, str], int] = {}
    out: list[dict] = []
    for pkg in packages:
        key = (pkg.get("name", ""), pkg.get("version", ""))
        if key not in seen:
            seen[key] = len(out)
            out.append(pkg)
            continue
        existing = out[seen[key]]
        existing_lic = existing.get("license") or "UNKNOWN"
        new_lic = pkg.get("license") or "UNKNOWN"
        if existing_lic == "UNKNOWN" and new_lic != "UNKNOWN":
            out[seen[key]] = pkg
    return out
