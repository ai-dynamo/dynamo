# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for license fetcher helpers (no network calls)."""

from __future__ import annotations

import base64
import json

import pytest
from dynamo_attributions.licenses import (
    LicenseCache,
    PackageLicense,
    _github_owner_repo_from_go_module,
    fetch_go_license,
    fetch_rust_license,
)


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.calls: list[tuple[str, dict | None]] = []

    def get(self, url, headers=None, timeout=None):  # noqa: D401
        self.calls.append((url, headers))
        return self._response


# ---------- _github_owner_repo_from_go_module ----------


@pytest.mark.parametrize(
    "module,expected",
    [
        ("github.com/foo/bar", ("foo", "bar")),
        ("github.com/foo/bar/v3", ("foo", "bar")),
        ("k8s.io/api", ("kubernetes", "api")),
        ("sigs.k8s.io/controller-runtime", ("kubernetes-sigs", "controller-runtime")),
        ("go.uber.org/zap", ("uber-go", "zap")),
        ("golang.org/x/sync", ("golang", "sync")),
        ("google.golang.org/grpc", ("grpc", "grpc-go")),
    ],
)
def test_github_owner_repo_known_mappings(module, expected):
    assert _github_owner_repo_from_go_module(module) == expected


def test_github_owner_repo_unknown_returns_none():
    assert _github_owner_repo_from_go_module("example.com/unknown/mod") is None


# ---------- fetch_go_license ----------


def test_fetch_go_license_decodes_base64_text():
    """License text returned base64-encoded should be decoded transparently."""
    text = "MIT License\n...\n"
    payload = {
        "license": {"spdx_id": "MIT"},
        "html_url": "https://github.com/foo/bar/blob/main/LICENSE",
        "encoding": "base64",
        "content": base64.b64encode(text.encode()).decode(),
    }
    session = _FakeSession(_FakeResponse(200, payload))
    pkg = fetch_go_license("github.com/foo/bar", "v1.0.0", session)

    assert pkg.license_expression == "MIT"
    assert pkg.license_text == text
    assert pkg.error == ""
    assert pkg.repository == "https://github.com/foo/bar"


def test_fetch_go_license_handles_404():
    session = _FakeSession(_FakeResponse(404, {}))
    pkg = fetch_go_license("github.com/foo/missing", "v0.1.0", session)
    assert pkg.error == "License not detected by GitHub"
    assert pkg.repository == "https://github.com/foo/missing"


def test_fetch_go_license_handles_invalid_base64():
    """Finding #16 — a corrupt content blob should produce empty license_text, not crash."""
    payload = {
        "license": {"spdx_id": "MIT"},
        "html_url": "https://github.com/foo/bar/blob/main/LICENSE",
        "encoding": "base64",
        "content": "!!!not-base64!!!",
    }
    session = _FakeSession(_FakeResponse(200, payload))
    pkg = fetch_go_license("github.com/foo/bar", "v1.0.0", session)
    assert pkg.license_expression == "MIT"
    assert pkg.license_text == ""


def test_fetch_go_license_no_github_mapping():
    session = _FakeSession(_FakeResponse(200, {}))
    pkg = fetch_go_license("example.com/unknown/mod", "v0.1.0", session)
    assert pkg.error == "No GitHub mapping for this module"
    assert session.calls == []  # never hit the API


# ---------- fetch_rust_license ----------


def test_fetch_rust_license_extracts_license_and_repo():
    payload = {
        "version": {"license": "MIT OR Apache-2.0"},
        "crate": {
            "repository": "https://github.com/foo/bar",
            "homepage": "https://example.com",
        },
    }
    session = _FakeSession(_FakeResponse(200, payload))
    pkg = fetch_rust_license("foo", "1.2.3", session)

    assert pkg.license_expression == "MIT OR Apache-2.0"
    assert pkg.repository == "https://github.com/foo/bar"
    assert pkg.license_url == "https://github.com/foo/bar/blob/HEAD/LICENSE"


def test_fetch_rust_license_handles_404():
    session = _FakeSession(_FakeResponse(404, {}))
    pkg = fetch_rust_license("nonexistent-crate", "0.0.1", session)
    assert pkg.error == "Not found on crates.io"


# ---------- LicenseCache ----------


def test_license_cache_roundtrip(tmp_path):
    cache_path = tmp_path / "cache.json"
    cache = LicenseCache(cache_path)
    cache.put(
        PackageLicense(
            name="serde",
            version="1.0.0",
            ecosystem="cargo",
            license_expression="MIT OR Apache-2.0",
        )
    )
    cache.save()

    raw = json.loads(cache_path.read_text())
    assert "cargo:serde:1.0.0" in raw

    fresh = LicenseCache(cache_path)
    hit = fresh.get("cargo", "serde", "1.0.0")
    assert hit is not None
    assert hit.license_expression == "MIT OR Apache-2.0"


def test_license_cache_skips_error_entries(tmp_path):
    cache_path = tmp_path / "cache.json"
    cache = LicenseCache(cache_path)
    cache.put(
        PackageLicense(
            name="busted",
            version="0.0.0",
            ecosystem="golang",
            error="some failure",
        )
    )
    cache.save()
    fresh = LicenseCache(cache_path)
    assert fresh.get("golang", "busted", "0.0.0") is None


def test_license_cache_recovers_from_corrupt_file(tmp_path):
    cache_path = tmp_path / "cache.json"
    cache_path.write_text("{not json")
    cache = LicenseCache(cache_path)
    assert cache.size == 0
