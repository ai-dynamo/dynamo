# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Causal cache provenance rejects unknown or modified released source."""

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dynamo.vllm import decision_cache_provenance as cache
from dynamo.vllm import decision_runtime_policy as policy
from dynamo.vllm.tests.decision_cache_contract import CONTRACT_PATH, project_contract

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.parametrize(
    "data,expected",
    [
        (b"", "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"),
        (b"hello", "b6fc4c620b67d95f953a5c1c1230aaab5db5a1b0"),
    ],
)
def test_git_blob_id_matches_known_git_objects(data, expected):
    assert cache.git_blob_id(data) == expected


@pytest.mark.parametrize("mode", ["valid", "changed", "missing"])
def test_shared_installed_source_verifier_checks_every_blob(tmp_path, mode):
    (tmp_path / "first.py").write_bytes(b"hello")
    if mode != "missing":
        (tmp_path / "second.py").write_bytes(b"hello" if mode == "changed" else b"")
    blobs = (
        ("first.py", "b6fc4c620b67d95f953a5c1c1230aaab5db5a1b0"),
        ("second.py", "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"),
    )
    assert cache.verify_installed_sources(blobs, root=tmp_path) is (mode == "valid")


@pytest.mark.parametrize("explicit_root", [False, True])
def test_malformed_installed_source_path_fails_closed(monkeypatch, explicit_root):
    monkeypatch.setattr(
        cache, "import_module", lambda _: SimpleNamespace(__file__="\0")
    )
    root = Path("\0") if explicit_root else None
    assert not cache.verify_installed_sources((("source.py", "unused"),), root=root)


@pytest.mark.parametrize("release", tuple(cache.CAUSAL_CACHE_SOURCE_BLOBS))
@pytest.mark.parametrize("mode", ["valid", "changed", "missing", "unknown", "no-path"])
def test_causal_cache_source_gate_is_exact_and_fail_closed(
    tmp_path, monkeypatch, release, mode
):
    data = b"# public scheduler fixture\n"
    blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    monkeypatch.setattr(
        cache, "CAUSAL_CACHE_SOURCE_BLOBS", {release: (("vllm/scheduler.py", blob),)}
    )
    package = tmp_path / "vllm"
    package.mkdir()
    if mode != "missing":
        (package / "scheduler.py").write_bytes(
            data if mode != "changed" else data + b"# modified\n"
        )
    monkeypatch.setattr(
        cache,
        "import_module",
        lambda _: SimpleNamespace(
            __file__=None if mode == "no-path" else str(package / "__init__.py")
        ),
    )
    actual = "0.99.0" if mode == "unknown" else release + "+cu130"
    assert cache.qualified_cache_runtime(actual) is (mode == "valid")


@pytest.mark.parametrize("error", [ImportError, OSError, RuntimeError, ValueError])
def test_unavailable_optional_cache_source_does_not_break_generation(
    monkeypatch, error
):
    def unavailable(_):
        raise error("optional source unavailable")

    monkeypatch.setattr(cache, "import_module", unavailable)
    assert not cache.qualified_cache_runtime("0.32.0")


@pytest.mark.parametrize("installed", ["invalid", None, 1])
def test_invalid_version_never_imports_backend_source(monkeypatch, installed):
    monkeypatch.setattr(
        cache, "import_module", lambda _: pytest.fail("Unexpected import")
    )
    assert not cache.qualified_cache_runtime(installed)


@pytest.mark.parametrize("installed", [None, 1])
def test_nonstring_cache_release_never_calls_version_parser(monkeypatch, installed):
    parser = Mock(side_effect=TypeError("version must be a string"))
    monkeypatch.setattr(policy, "Version", parser)
    monkeypatch.setattr(cache, "Version", parser, raising=False)
    assert not cache.qualified_cache_runtime(installed)
    parser.assert_not_called()


def test_released_cache_provenance_contracts_are_identical():
    contract = json.loads(CONTRACT_PATH.read_text())
    releases = contract["releases"]
    assert releases.keys() == cache.CAUSAL_CACHE_SOURCE_BLOBS.keys()
    for release, projection in releases.items():
        assert projection["commit"] == cache.CAUSAL_RELEASE_COMMITS[release]
        assert projection["blobs"] == dict(cache.CAUSAL_CACHE_SOURCE_BLOBS[release])
    assert releases["0.32.0"]["projection"] == releases["0.33.0"]["projection"]
    repository = os.environ.get("DYNAMO_VLLM_SOURCE_REPO")
    if repository is not None:
        assert project_contract(repository) == contract
