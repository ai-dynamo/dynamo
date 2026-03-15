# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""License fetcher for Rust and Go transitive dependencies.

Queries registry APIs (crates.io for Rust, GitHub for Go) to retrieve license
metadata for resolved packages, with persistent JSON caching and rate limiting.
Python packages are handled via --image (container inspection), not this module.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from .types import Ecosystem, ResolvedPackage

logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = Path.home() / ".dynamo_license_cache.json"

_RATE_LIMITS: dict[Ecosystem, float] = {
    Ecosystem.RUST: 1.0,
    Ecosystem.PYTHON: 0.0,  # Python uses --image, not API calls
    Ecosystem.GO: 0.0,
}


@dataclass
class PackageLicense:
    """License metadata for a single resolved package."""

    name: str
    version: str
    ecosystem: str
    license_expression: str = ""
    repository: str = ""
    homepage: str = ""
    license_text: str = ""
    license_url: str = ""
    project_urls: dict[str, str] = field(default_factory=dict)
    error: str = ""

    def to_resolver_dict(self) -> dict[str, Any]:
        """Convert to the dict format expected by the renderers."""
        d: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "ecosystem": self.ecosystem,
            "license_expression": self.license_expression,
            "repository": self.repository,
            "homepage": self.homepage,
        }
        if self.license_text:
            d["license_text"] = self.license_text
        if self.license_url:
            d["license_urls"] = [self.license_url]
        if self.project_urls:
            d["project_urls"] = self.project_urls
        return d


_PKG_LICENSE_FIELDS = {f.name for f in fields(PackageLicense)}


class LicenseCache:
    """JSON file-backed cache keyed by ecosystem:name:version."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or DEFAULT_CACHE_PATH
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning(
                    "Corrupt license cache at %s, starting fresh", self._path
                )
                self._data = {}

    def save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._data, indent=1, sort_keys=True), encoding="utf-8"
            )
        except OSError as exc:
            logger.warning("Could not save license cache to %s: %s", self._path, exc)

    @staticmethod
    def _key(ecosystem: str, name: str, version: str) -> str:
        return f"{ecosystem}:{name}:{version}"

    def get(self, ecosystem: str, name: str, version: str) -> PackageLicense | None:
        key = self._key(ecosystem, name, version)
        entry = self._data.get(key)
        if entry is None:
            return None
        if entry.get("error"):
            return None
        valid = {
            k: v for k, v in entry.items() if k in _PKG_LICENSE_FIELDS and k != "error"
        }
        return PackageLicense(**valid, error="")

    def put(self, pkg: PackageLicense) -> None:
        key = self._key(pkg.ecosystem, pkg.name, pkg.version)
        self._data[key] = asdict(pkg)

    @property
    def size(self) -> int:
        return len(self._data)


_SESSION_HEADERS = {"User-Agent": "dynamo-attributions/1.0"}


def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(_SESSION_HEADERS)
    return session


def fetch_rust_license(
    name: str,
    version: str,
    session: requests.Session,
) -> PackageLicense:
    """Fetch license info for a Rust crate from crates.io."""
    url = f"https://crates.io/api/v1/crates/{quote(name, safe='')}/{quote(version, safe='')}"
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code == 404:
            return PackageLicense(
                name=name,
                version=version,
                ecosystem="cargo",
                error="Not found on crates.io",
            )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        return PackageLicense(
            name=name,
            version=version,
            ecosystem="cargo",
            error=str(exc),
        )

    ver_data = data.get("version", {})
    crate_data = data.get("crate", {})

    license_expr = ver_data.get("license") or ""
    repository = crate_data.get("repository") or ver_data.get("repository") or ""
    homepage = crate_data.get("homepage") or ""

    license_url = ""
    if repository and "github.com" in repository:
        license_url = repository.rstrip("/") + "/blob/HEAD/LICENSE"

    return PackageLicense(
        name=name,
        version=version,
        ecosystem="cargo",
        license_expression=license_expr,
        repository=repository,
        homepage=homepage,
        license_url=license_url,
    )


_GO_VANITY_TO_GITHUB: list[tuple[str, str, str]] = [
    ("golang.org/x/", "golang", "{name}"),
    ("google.golang.org/grpc", "grpc", "grpc-go"),
    ("google.golang.org/protobuf", "protocolbuffers", "protobuf-go"),
    ("google.golang.org/genproto", "googleapis", "go-genproto"),
    ("google.golang.org/api", "googleapis", "google-api-go-client"),
    ("k8s.io/", "kubernetes", "{name}"),
    ("sigs.k8s.io/", "kubernetes-sigs", "{name}"),
    ("go.uber.org/", "uber-go", "{name}"),
    ("go.opentelemetry.io/otel", "open-telemetry", "opentelemetry-go"),
    ("go.opentelemetry.io/contrib", "open-telemetry", "opentelemetry-go-contrib"),
    ("go.opentelemetry.io/auto", "open-telemetry", "opentelemetry-go-instrumentation"),
    ("go.opentelemetry.io/proto", "open-telemetry", "opentelemetry-proto-go"),
    ("gopkg.in/yaml.v", "go-yaml", "yaml"),
    ("gopkg.in/inf.v", "go-inf", "inf"),
    ("gopkg.in/evanphx/json-patch.v", "evanphx", "json-patch"),
    ("istio.io/api", "istio", "api"),
    ("istio.io/client-go", "istio", "client-go"),
    ("volcano.sh/apis", "volcano-sh", "apis"),
    ("cel.dev/expr", "google", "cel-spec"),
    ("emperror.dev/errors", "emperror", "errors"),
    ("gomodules.xyz/jsonpatch", "gomodules", "jsonpatch"),
    ("go.yaml.in/yaml", "go-yaml", "yaml"),
]


def _github_owner_repo_from_go_module(module: str) -> tuple[str, str] | None:
    """Extract GitHub owner/repo from a Go module path."""
    match = re.match(r"github\.com/([^/]+)/([^/]+)", module)
    if match:
        return match.group(1), match.group(2)

    for prefix, owner, repo_template in _GO_VANITY_TO_GITHUB:
        if not module.startswith(prefix):
            continue
        if "{name}" in repo_template:
            remainder = module[len(prefix) :]
            name = remainder.split("/")[0]
            if re.match(r"^v\d+$", name):
                continue
            return owner, repo_template.replace("{name}", name)
        return owner, repo_template

    return None


def fetch_go_license(
    module: str,
    version: str,
    session: requests.Session,
    github_token: str | None = None,
) -> PackageLicense:
    """Fetch license info for a Go module via GitHub Licenses API."""
    owner_repo = _github_owner_repo_from_go_module(module)
    if not owner_repo:
        return PackageLicense(
            name=module,
            version=version,
            ecosystem="golang",
            license_expression="Unknown",
            repository=f"https://{module}",
            error="No GitHub mapping for this module",
        )

    owner, repo = owner_repo
    api_url = f"https://api.github.com/repos/{owner}/{repo}/license"
    headers: dict[str, str] = {}
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    try:
        resp = session.get(api_url, headers=headers, timeout=15)
        if resp.status_code == 404:
            return PackageLicense(
                name=module,
                version=version,
                ecosystem="golang",
                repository=f"https://github.com/{owner}/{repo}",
                error="License not detected by GitHub",
            )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        return PackageLicense(
            name=module,
            version=version,
            ecosystem="golang",
            repository=f"https://github.com/{owner}/{repo}",
            error=str(exc),
        )

    license_info = data.get("license", {})
    spdx_id = license_info.get("spdx_id") or "Unknown"
    html_url = data.get("html_url", "")
    license_text = data.get("content", "")
    if data.get("encoding") == "base64" and license_text:
        try:
            license_text = base64.b64decode(license_text).decode(
                "utf-8", errors="replace"
            )
        except Exception:
            license_text = ""

    return PackageLicense(
        name=module,
        version=version,
        ecosystem="golang",
        license_expression=spdx_id,
        repository=f"https://github.com/{owner}/{repo}",
        license_text=license_text,
        license_url=html_url,
    )


_ECO_API_LABEL: dict[Ecosystem, str] = {
    Ecosystem.RUST: "cargo",
    Ecosystem.PYTHON: "pypi",
    Ecosystem.GO: "golang",
}


def _fetch_one(
    pkg: ResolvedPackage,
    session: requests.Session,
    github_token: str | None,
) -> PackageLicense:
    """Dispatch a single license lookup to the appropriate ecosystem fetcher."""
    if pkg.ecosystem == Ecosystem.RUST:
        return fetch_rust_license(pkg.name, pkg.version, session)
    return fetch_go_license(pkg.name, pkg.version, session, github_token)


def _apply_rate_limit(
    ecosystem: Ecosystem,
    last_request_time: dict[Ecosystem, float],
) -> None:
    """Sleep if needed to respect per-ecosystem rate limits."""
    rate_limit = _RATE_LIMITS.get(ecosystem, 0.5)
    if rate_limit <= 0:
        return
    last = last_request_time.get(ecosystem, 0.0)
    elapsed = time.time() - last
    if elapsed < rate_limit:
        time.sleep(rate_limit - elapsed)


def fetch_all_licenses(
    packages: list[ResolvedPackage],
    cache_path: Path | None = None,
    github_token: str | None = None,
    on_progress: Any | None = None,
) -> list[PackageLicense]:
    """Fetch license metadata for all packages, using cache where possible."""
    cache = LicenseCache(cache_path)
    session = _make_session()
    results: list[PackageLicense] = []
    last_request_time: dict[Ecosystem, float] = {}
    cache_hits = 0
    total = len(packages)

    for i, pkg in enumerate(packages):
        api_label = _ECO_API_LABEL[pkg.ecosystem]

        cached = cache.get(api_label, pkg.name, pkg.version)
        if cached:
            results.append(cached)
            cache_hits += 1
        else:
            _apply_rate_limit(pkg.ecosystem, last_request_time)
            result = _fetch_one(pkg, session, github_token)
            last_request_time[pkg.ecosystem] = time.time()
            if not result.error:
                cache.put(result)
            results.append(result)

        if on_progress:
            on_progress(i + 1, total, pkg.name)

    cache.save()
    logger.info(
        "License fetch complete: %d total, %d cache hits, %d API calls",
        total,
        cache_hits,
        total - cache_hits,
    )
    return results
