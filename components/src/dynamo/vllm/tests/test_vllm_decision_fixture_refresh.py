# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline fixture refresh verifies every selected source before replacing files."""

import sys
from unittest.mock import Mock

import pytest

from dynamo.vllm.tests import decision_stub_contract as generator

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]


def test_bundled_check_cli_uses_only_current_source_subset(monkeypatch):
    root = generator.BUNDLED_SOURCE_ROOT
    files = tuple(sorted(root.rglob("*.py.txt")))
    expected = {
        path.relative_to(root).as_posix()[:-4]: path.read_text() for path in files
    }
    assert len(expected) == 7
    reader = Mock(side_effect=lambda repo, path: expected[path])
    monkeypatch.setattr(generator, "read_pinned_source", reader)
    monkeypatch.setattr(
        sys, "argv", ["generator", "--source-repo", "offline", "--check-bundled"]
    )
    generator.main()
    assert {call.args[1] for call in reader.call_args_list} == expected.keys()
    reader.side_effect = lambda repo, path: expected[path] + "# changed\n"
    with pytest.raises(AssertionError, match="stale"):
        generator.main()
    assert all(
        path.read_text() == expected[path.relative_to(root).as_posix()[:-4]]
        for path in files
    )


def _bundle(tmp_path, monkeypatch):
    files = tuple(tmp_path / name for name in ("first.py.txt", "second.py.txt"))
    for path in files:
        path.write_bytes(b"old\n")
    monkeypatch.setattr(generator, "BUNDLED_SOURCE_ROOT", tmp_path)
    return files


def test_refresh_resolves_all_sources_before_replacing_any(monkeypatch, tmp_path):
    files = _bundle(tmp_path, monkeypatch)

    def read(repository, path):
        assert all(file.read_bytes() == b"old\n" for file in files)
        return f"# {path} λ\n"

    monkeypatch.setattr(generator, "read_pinned_source", read)
    assert generator.refresh_bundled_sources("offline", check=False) == 2
    assert [path.read_text() for path in files] == ["# first.py λ\n", "# second.py λ\n"]
    assert tuple(sorted(tmp_path.iterdir())) == files


def test_failed_source_verification_does_not_partially_refresh(monkeypatch, tmp_path):
    files = _bundle(tmp_path, monkeypatch)
    reader = Mock(
        side_effect=["# verified first\n", AssertionError("pinned blob mismatch")]
    )
    monkeypatch.setattr(generator, "read_pinned_source", reader)
    with pytest.raises(AssertionError, match="pinned blob mismatch"):
        generator.refresh_bundled_sources("offline", check=False)
    assert all(path.read_bytes() == b"old\n" for path in files)


def test_unrecognized_bundled_source_fails_closed(monkeypatch, tmp_path):
    files = _bundle(tmp_path, monkeypatch)
    with pytest.raises(KeyError):
        generator.refresh_bundled_sources("not-an-accessed-git-repository", check=False)
    assert all(path.read_bytes() == b"old\n" for path in files)
