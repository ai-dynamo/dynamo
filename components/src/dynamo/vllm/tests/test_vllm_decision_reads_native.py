# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned source fixture loading accepts bare and ordinary Git repositories."""

import ast
import subprocess

import pytest

from dynamo.vllm import decision_native_loader as contract
from dynamo.vllm.tests import decision_pinned_source as pinned_source
from dynamo.vllm.tests.decision_pinned_source import (
    read_pinned_source,
    source_repository,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]


@pytest.mark.parametrize("strict", [False, True])
def test_bundled_cpu_sources_do_not_require_repository_configuration(
    monkeypatch, strict
):
    monkeypatch.delenv("DYNAMO_VLLM_SOURCE_REPO", raising=False)
    monkeypatch.setenv("DYNAMO_VLLM_REQUIRE_PINNED_SOURCE", "1" if strict else "0")
    try:
        repository = source_repository()
    except (pytest.skip.Exception, pytest.fail.Exception) as cause:
        pytest.fail(
            f"Mandatory native CPU fixtures must run without an override: {cause}"
        )
    assert repository is None
    for path in (
        "vllm/entrypoints/openai/decisions/strategies.py",
        "vllm/entrypoints/openai/decisions/question_types.py",
        "vllm/entrypoints/openai/decisions/adapters.py",
        "vllm/entrypoints/generate/label_reads.py",
        "vllm/utils/async_utils.py",
        "vllm/exceptions.py",
        "vllm/v1/engine/exceptions.py",
    ):
        assert ast.parse(read_pinned_source(repository, path)).body


def test_modified_bundled_source_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(pinned_source, "BUNDLED_SOURCE_ROOT", tmp_path)
    source = tmp_path / "vllm/exceptions.py.txt"
    source.parent.mkdir()
    source.write_text("modified fixture", encoding="utf-8")
    with pytest.raises(AssertionError, match="pinned Git blob"):
        read_pinned_source(None, "vllm/exceptions.py")


def test_missing_bundled_source_does_not_skip_or_fall_back(tmp_path, monkeypatch):
    monkeypatch.setattr(pinned_source, "BUNDLED_SOURCE_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        read_pinned_source(None, "vllm/exceptions.py")


def test_explicit_repository_override_never_falls_back(tmp_path, monkeypatch):
    repository = tmp_path / "missing-upstream"
    monkeypatch.setenv("DYNAMO_VLLM_SOURCE_REPO", str(repository))
    assert source_repository() == str(repository)
    with pytest.raises(subprocess.CalledProcessError):
        read_pinned_source(source_repository(), "vllm/exceptions.py")


def test_empty_repository_override_fails_closed(monkeypatch):
    monkeypatch.setenv("DYNAMO_VLLM_SOURCE_REPO", "")
    with pytest.raises(ValueError, match="must be nonempty"):
        source_repository()


@pytest.mark.parametrize("bare", [False, True])
def test_source_loader_accepts_bare_and_worktree_paths(tmp_path, monkeypatch, bare):
    repository = tmp_path / "upstream"
    subprocess.run(["git", "init", str(repository)], check=True, capture_output=True)
    data = b"fixture = True\n"
    blob = (
        subprocess.check_output(
            ["git", "-C", str(repository), "hash-object", "-w", "--stdin"], input=data
        )
        .decode()
        .strip()
    )
    tree = (
        subprocess.check_output(
            ["git", "-C", str(repository), "mktree"],
            input=f"100644 blob {blob}\tfixture.py\n".encode(),
        )
        .decode()
        .strip()
    )
    commit = (
        subprocess.check_output(
            [
                "git",
                "-C",
                str(repository),
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit-tree",
                tree,
            ],
            input=b"fixture\n",
        )
        .decode()
        .strip()
    )
    subprocess.run(
        ["git", "-C", str(repository), "update-ref", "refs/heads/main", commit],
        check=True,
    )
    if bare:
        destination = tmp_path / "bare.git"
        subprocess.run(
            ["git", "clone", "--bare", str(repository), str(destination)],
            check=True,
            capture_output=True,
        )
        repository = destination
    monkeypatch.setattr(contract, "PINNED_VLLM_COMMIT", commit)
    monkeypatch.setattr(contract, "PINNED_SOURCE_BLOBS", (("fixture.py", blob),))
    assert read_pinned_source(repository, "fixture.py") == data.decode()
