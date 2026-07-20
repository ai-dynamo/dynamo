# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provenance and subtraction checks for the pinned vLLM nightly baseline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from compliance.generators.common import (
    Component,
    load_subtract_keys,
    subtract_baseline,
)
from container.compliance.base_sboms import capture_baseline_sbom
from container.compliance.base_sboms import check_drift

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

_REPO = Path(__file__).resolve().parents[3]
_BASE_SBOM_DIR = _REPO / "container/compliance/base_sboms"
_SBOM_STEM = "vllm-openai@7f2bc168"
_SBOM = _BASE_SBOM_DIR / f"{_SBOM_STEM}-amd64.cdx.json"
_INDEX_DIGEST = (
    "sha256:7f2bc168366c77fbd8329368f00310d208531c14ece6c2de31a6611ef99f6ec8"
)
_AMD64_DIGEST = (
    "sha256:99e7dd3cf74c489af0615671f3fdbde182de2930f1195a0ee39e914e38033a88"
)
_TAG = "nightly-dcfebf93f4eccf30f71872283331eee757915daf"


def _baseline_entry() -> dict:
    manifest = json.loads((_BASE_SBOM_DIR / "manifest.json").read_text())
    return next(
        entry
        for entry in manifest["entries"]
        if entry.get("baseline_sbom") == _SBOM.name
    )


def test_manifest_binds_sbom_to_exact_index_and_amd64_manifest():
    entry = _baseline_entry()

    assert entry["from_image"] == entry["baseline_image"] == "vllm/vllm-openai"
    assert entry["from_tag"] == entry["baseline_tag"] == _TAG
    assert entry["from_digest"] == entry["baseline_digest"] == _INDEX_DIGEST
    assert entry["baseline_platform_digest"] == _AMD64_DIGEST
    assert entry["platform"] == "linux/amd64"

    sbom = json.loads(_SBOM.read_text())
    component = sbom["metadata"]["component"]
    assert component["name"] == "vllm/vllm-openai"
    assert component["version"] == _AMD64_DIGEST


def test_platform_digest_resolves_the_index_child(monkeypatch):
    index = {
        "manifests": [
            {
                "digest": "sha256:arm64",
                "platform": {"os": "linux", "architecture": "arm64"},
            },
            {
                "digest": _AMD64_DIGEST,
                "platform": {"os": "linux", "architecture": "amd64"},
            },
        ]
    }
    monkeypatch.setattr(
        capture_baseline_sbom,
        "_imagetools_raw",
        lambda _ref: json.dumps(index).encode(),
    )

    assert (
        capture_baseline_sbom.resolve_platform_digest(
            "example.invalid/image:tag", "linux/amd64"
        )
        == _AMD64_DIGEST
    )
    with pytest.raises(ValueError, match="no platform linux/s390x"):
        capture_baseline_sbom.resolve_platform_digest(
            "example.invalid/image:tag", "linux/s390x"
        )


def test_drift_check_fails_on_platform_manifest_mismatch(monkeypatch):
    entry = _baseline_entry()
    monkeypatch.setattr(check_drift, "resolve_index_digest", lambda _ref: _INDEX_DIGEST)
    monkeypatch.setattr(
        check_drift, "resolve_platform_digest", lambda _ref, _platform: "sha256:changed"
    )
    monkeypatch.setattr(
        check_drift, "resolve_platform_layers", lambda _ref, _platform: []
    )

    problems = check_drift.check_entry(entry)

    assert len(problems) == 1
    assert "PLATFORM-MANIFEST DRIFT" in problems[0]


def test_subtraction_keeps_overlay_additions_visible():
    baseline_keys = load_subtract_keys(_SBOM)
    inherited = {
        ("bash", "5.1-6ubuntu1.1"),
        ("coreutils", "8.32-4.1ubuntu1.2"),
        ("cuda-keyring", "1.1-1"),
        ("libpython3.12", "3.12.13-1+jammy1"),
        ("nvidia-cutlass-dsl-libs-core", "4.6.0"),
        ("flashinfer-python", "0.6.14"),
    }
    assert inherited <= baseline_keys

    components = [
        Component("dpkg", name, version, "UNKNOWN") for name, version in inherited
    ]
    components.extend(
        [
            Component("python", "flashinfer-python", "0.6.15", "Apache-2.0"),
            Component("python", "flashinfer-cubin", "0.6.15", "Apache-2.0"),
            Component("dpkg", "cuda-nvrtc-dev-13-0", "13.0.88-1", "UNKNOWN"),
        ]
    )

    kept = {(c.name, c.version) for c in subtract_baseline(components, baseline_keys)}
    assert kept == {
        ("flashinfer-python", "0.6.15"),
        ("flashinfer-cubin", "0.6.15"),
        ("cuda-nvrtc-dev-13-0", "13.0.88-1"),
    }
