# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline unit tests for the license-lookup module.

These tests never touch the network. ``_http_json`` is monkey-patched so the
crates.io and deps.dev paths are exercised against fixed payloads, and the
SQLite cache is redirected to a tempfile so the user's real cache is untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from container.compliance.lockfile import license_lookup


@pytest.fixture(autouse=True)
def _reset_throttle() -> None:
    """Clear the per-host throttle so tests don't sleep between runs."""
    license_lookup._last_call.clear()


def _patch_http(monkeypatch: pytest.MonkeyPatch, payloads: dict[str, Any]) -> list[str]:
    """Patch _http_json to return canned payloads keyed by URL fragment.

    Returns a mutable list that records every URL the resolver tried, so tests
    can assert on call counts (cache hit vs miss).
    """
    calls: list[str] = []

    def fake(url: str, *, headers: dict[str, str] | None = None) -> Any:
        calls.append(url)
        for fragment, payload in payloads.items():
            if fragment in url:
                return payload
        return None

    monkeypatch.setattr(license_lookup, "_http_json", fake)
    return calls


# ---------- crates.io ----------


def test_fetch_cargo_license_returns_spdx(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_http(
        monkeypatch,
        {"/api/v1/crates/serde/1.0.0": {"version": {"license": "MIT OR Apache-2.0"}}},
    )
    assert license_lookup.fetch_cargo_license("serde", "1.0.0") == "MIT OR Apache-2.0"


def test_fetch_cargo_license_handles_missing_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_http(monkeypatch, {})
    assert license_lookup.fetch_cargo_license("does-not-exist", "9.9.9") == "UNKNOWN"


def test_fetch_cargo_license_handles_blank_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_http(
        monkeypatch,
        {"/api/v1/crates/blank/0.1.0": {"version": {"license": "   "}}},
    )
    assert license_lookup.fetch_cargo_license("blank", "0.1.0") == "UNKNOWN"


def test_fetch_cargo_license_url_encodes_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_http(
        monkeypatch,
        {"weird%2Fname": {"version": {"license": "MIT"}}},
    )
    assert license_lookup.fetch_cargo_license("weird/name", "1.0.0") == "MIT"
    assert calls and "weird%2Fname" in calls[0]


# ---------- deps.dev ----------


def test_fetch_go_license_joins_multiple(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_http(
        monkeypatch,
        {"/systems/go/packages/": {"licenses": ["Apache-2.0", "MIT"]}},
    )
    assert (
        license_lookup.fetch_go_license("github.com/foo/bar", "v1.2.3")
        == "Apache-2.0 AND MIT"
    )


def test_fetch_go_license_url_encodes_module_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_http(
        monkeypatch,
        {"github.com%2Ffoo%2Fbar": {"licenses": ["MIT"]}},
    )
    license_lookup.fetch_go_license("github.com/foo/bar", "v0.1.0")
    assert calls and "github.com%2Ffoo%2Fbar" in calls[0]


def test_fetch_go_license_returns_unknown_on_empty_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_http(monkeypatch, {"/systems/go/packages/": {"licenses": []}})
    assert license_lookup.fetch_go_license("github.com/foo/bar", "v0.0.1") == "UNKNOWN"


# ---------- PyPI ----------


def test_fetch_pypi_license_prefers_license_expression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_http(
        monkeypatch,
        {
            "/pypi/serde/1.0.0/json": {
                "info": {"license_expression": "MIT OR Apache-2.0"}
            }
        },
    )
    assert license_lookup.fetch_pypi_license("serde", "1.0.0") == "MIT OR Apache-2.0"


def test_fetch_pypi_license_falls_back_to_classifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When license_expression is missing, classifiers are the next-best signal."""
    _patch_http(
        monkeypatch,
        {
            "/pypi/requests/2.31.0/json": {
                "info": {
                    "license": "Apache 2.0\n\nCopyright 2024 ...",
                    "classifiers": [
                        "License :: OSI Approved :: Apache Software License",
                        "Programming Language :: Python :: 3",
                    ],
                }
            }
        },
    )
    assert license_lookup.fetch_pypi_license("requests", "2.31.0") == "Apache-2.0"


def test_fetch_pypi_license_joins_multiple_classifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_http(
        monkeypatch,
        {
            "/pypi/multilic/1.0/json": {
                "info": {
                    "classifiers": [
                        "License :: OSI Approved :: MIT License",
                        "License :: OSI Approved :: Apache Software License",
                    ]
                }
            }
        },
    )
    assert license_lookup.fetch_pypi_license("multilic", "1.0") == "Apache-2.0 AND MIT"


def test_fetch_pypi_license_uses_short_license_field_as_last_resort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_http(
        monkeypatch,
        {"/pypi/foo/1/json": {"info": {"license": "BSD-3-Clause"}}},
    )
    assert license_lookup.fetch_pypi_license("foo", "1") == "BSD-3-Clause"


def test_fetch_pypi_license_rejects_long_license_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A multi-paragraph LICENSE-file dump must not be returned verbatim."""
    _patch_http(
        monkeypatch,
        {
            "/pypi/foo/1/json": {
                "info": {"license": "Copyright 2024 Foo Inc. All rights reserved."}
            }
        },
    )
    assert license_lookup.fetch_pypi_license("foo", "1") == "UNKNOWN"


def test_fetch_pypi_license_falls_back_to_github(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When PyPI metadata is empty, follow project_urls to GitHub's license API."""
    _patch_http(
        monkeypatch,
        {
            "/pypi/setuptools/79.0.1/json": {
                "info": {
                    "project_urls": {
                        "Source": "https://github.com/pypa/setuptools",
                    }
                }
            },
            "/repos/pypa/setuptools/license": {
                "license": {"spdx_id": "MIT"},
            },
        },
    )
    assert license_lookup.fetch_pypi_license("setuptools", "79.0.1") == "MIT"


def test_fetch_pypi_license_treats_github_NOASSERTION_as_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_http(
        monkeypatch,
        {
            "/pypi/foo/1/json": {
                "info": {"home_page": "https://github.com/owner/repo"}
            },
            "/repos/owner/repo/license": {"license": {"spdx_id": "NOASSERTION"}},
        },
    )
    assert license_lookup.fetch_pypi_license("foo", "1") == "UNKNOWN"


def test_fetch_pypi_license_strips_dot_git_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`.git` on the GitHub URL must not bleed into the API call."""
    calls = _patch_http(
        monkeypatch,
        {
            "/pypi/foo/1/json": {
                "info": {"home_page": "https://github.com/owner/repo.git"}
            },
            "/repos/owner/repo/license": {"license": {"spdx_id": "MIT"}},
        },
    )
    assert license_lookup.fetch_pypi_license("foo", "1") == "MIT"
    assert any("/repos/owner/repo/license" in c for c in calls)
    assert not any("repo.git" in c for c in calls)


def test_strip_pep440_local_drops_plus_segment() -> None:
    assert license_lookup._strip_pep440_local("2.7.4.post1+nv26.2") == "2.7.4.post1"
    assert license_lookup._strip_pep440_local("2.9.1+cu129") == "2.9.1"
    assert license_lookup._strip_pep440_local("1.0.0") == "1.0.0"


def test_fetch_pypi_license_retries_with_pep440_base_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 404 on a `+local` version must trigger a retry against the base."""
    calls = _patch_http(
        monkeypatch,
        {
            "/pypi/torchaudio/2.9.1/json": {
                "info": {
                    "classifiers": ["License :: OSI Approved :: BSD License"],
                },
            },
        },
    )
    assert license_lookup.fetch_pypi_license("torchaudio", "2.9.1+cu129") == "BSD-3-Clause"
    assert any("/pypi/torchaudio/2.9.1%2Bcu129/json" in c for c in calls)
    assert any("/pypi/torchaudio/2.9.1/json" in c for c in calls)


def test_fetch_pypi_license_does_not_retry_when_no_local_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plain versions must only hit PyPI once even on a 404."""
    calls = _patch_http(monkeypatch, {})
    assert license_lookup.fetch_pypi_license("missing", "1.2.3") == "UNKNOWN"
    assert sum(1 for c in calls if "/pypi/missing/" in c) == 1


# ---------- resolve_licenses + cache ----------


def test_resolve_licenses_caches_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A second resolve pass must not re-issue any HTTP calls."""
    cache_path = str(tmp_path / "cache.sqlite")
    calls = _patch_http(
        monkeypatch,
        {
            "/api/v1/crates/serde/1.0.0": {"version": {"license": "MIT OR Apache-2.0"}},
            "/api/v1/crates/anyhow/1.0.0": {"version": {"license": "Apache-2.0"}},
        },
    )
    entries = [
        {"name": "serde", "version": "1.0.0", "ecosystem": "cargo"},
        {"name": "anyhow", "version": "1.0.0", "ecosystem": "cargo"},
    ]

    first = license_lookup.resolve_licenses(entries, cache_path=cache_path)
    assert {(e["name"], e["license"]) for e in first} == {
        ("serde", "MIT OR Apache-2.0"),
        ("anyhow", "Apache-2.0"),
    }
    assert len(calls) == 2

    second = license_lookup.resolve_licenses(entries, cache_path=cache_path)
    assert second == first
    assert len(calls) == 2  # no new network calls


def test_resolve_licenses_preserves_existing_non_unknown_license(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If an entry already has a license, do not call the network at all."""
    cache_path = str(tmp_path / "cache.sqlite")
    calls = _patch_http(monkeypatch, {})
    entries = [
        {
            "name": "serde",
            "version": "1.0.0",
            "ecosystem": "cargo",
            "license": "MIT",
        }
    ]
    out = license_lookup.resolve_licenses(entries, cache_path=cache_path)
    assert out[0]["license"] == "MIT"
    assert calls == []


def test_resolve_licenses_retries_unknown_so_smarter_fetcher_wins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An UNKNOWN cached value must be treated as a miss and re-fetched.

    This protects against the common case where a fetcher gains a new
    fallback (e.g. PyPI -> GitHub) and we want existing runs to pick it up
    without manually invalidating the cache.
    """
    cache_path = str(tmp_path / "cache.sqlite")
    payloads: dict[str, Any] = {}
    calls = _patch_http(monkeypatch, payloads)
    entries = [{"name": "ghosty", "version": "1", "ecosystem": "cargo"}]

    # First run: empty payload -> UNKNOWN, must NOT be cached.
    license_lookup.resolve_licenses(entries, cache_path=cache_path)
    assert len(calls) == 1

    # Second run with a now-resolving payload: must re-query and resolve.
    payloads["/api/v1/crates/ghosty/1"] = {"version": {"license": "MIT"}}
    out = license_lookup.resolve_licenses(entries, cache_path=cache_path)
    assert out[0]["license"] == "MIT"
    assert len(calls) == 2

    # Third run: now MIT is cached, no further calls.
    out2 = license_lookup.resolve_licenses(entries, cache_path=cache_path)
    assert out2[0]["license"] == "MIT"
    assert len(calls) == 2


def test_resolve_licenses_unknown_ecosystem_short_circuits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_path = str(tmp_path / "cache.sqlite")
    calls = _patch_http(monkeypatch, {"anything": {"license": "MIT"}})
    out = license_lookup.resolve_licenses(
        [{"name": "x", "version": "1", "ecosystem": "ruby"}],
        cache_path=cache_path,
    )
    assert out[0]["license"] == "UNKNOWN"
    assert calls == []


# ---------- dedupe_packages ----------


def test_dedupe_prefers_non_unknown_license() -> None:
    rows = [
        {"name": "pyyaml", "version": "6.0.1", "license": "UNKNOWN"},
        {"name": "pyyaml", "version": "6.0.1", "license": "MIT"},
        {"name": "anyhow", "version": "1.0.0", "license": "Apache-2.0"},
    ]
    out = license_lookup.dedupe_packages(rows)
    assert len(out) == 2
    pyyaml = next(r for r in out if r["name"] == "pyyaml")
    assert pyyaml["license"] == "MIT"


def test_dedupe_preserves_first_seen_order() -> None:
    rows = [
        {"name": "b", "version": "1", "license": "MIT"},
        {"name": "a", "version": "1", "license": "MIT"},
        {"name": "b", "version": "1", "license": "MIT"},
    ]
    out = license_lookup.dedupe_packages(rows)
    assert [r["name"] for r in out] == ["b", "a"]


def test_dedupe_ignores_when_both_unknown() -> None:
    """Two UNKNOWN rows still collapse to a single UNKNOWN row."""
    rows = [
        {"name": "x", "version": "1", "license": "UNKNOWN"},
        {"name": "x", "version": "1", "license": "UNKNOWN"},
    ]
    out = license_lookup.dedupe_packages(rows)
    assert out == [{"name": "x", "version": "1", "license": "UNKNOWN"}]
