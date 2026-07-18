# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the exact GMS checkpoint cache-eviction helper."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]

HELPER_PATH = (
    Path(__file__).parents[2]
    / "benchmarks"
    / "gms_cuda_init_ab"
    / "gms-fadvise-exact.py"
)
RUNNER_PATH = HELPER_PATH.with_name("run-variant.sh")
SPEC = importlib.util.spec_from_file_location("gms_fadvise_exact", HELPER_PATH)
assert SPEC is not None and SPEC.loader is not None
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)


def test_reports_machine_readable_per_root_success(tmp_path, capsys):
    roots = [tmp_path / "checkpoint", tmp_path / "nvme"]
    sizes = [3, 5]
    for root, size in zip(roots, sizes):
        root.mkdir()
        (root / "block").write_bytes(b"x" * size)

    assert HELPER.main([str(root) for root in roots]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "ok": True,
        "roots": [
            {
                "root": str(root),
                "status": "ok",
                "files": 1,
                "bytes": size,
                "errors": 0,
            }
            for root, size in zip(roots, sizes)
        ],
        "total": {
            "roots": 2,
            "files": 2,
            "bytes": sum(sizes),
            "errors": 0,
        },
    }


def test_rejects_every_invalid_root(tmp_path, capsys):
    missing = tmp_path / "missing"
    not_directory = tmp_path / "checkpoint-file"
    not_directory.write_text("checkpoint", encoding="utf-8")
    empty = tmp_path / "empty"
    empty.mkdir()

    assert HELPER.main([str(missing), str(not_directory), str(empty)]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert [(result["status"], result["files"]) for result in payload["roots"]] == [
        ("missing", 0),
        ("not_directory", 0),
        ("empty", 0),
    ]


def test_rejects_per_root_eviction_error(tmp_path, capsys, monkeypatch):
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "block").write_bytes(b"checkpoint")

    def fail_fadvise(*_args):
        raise OSError("injected fadvise failure")

    monkeypatch.setattr(HELPER.os, "posix_fadvise", fail_fadvise)

    assert HELPER.main([str(root)]) == 1

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["roots"] == [
        {
            "root": str(root),
            "status": "error",
            "files": 0,
            "bytes": 0,
            "errors": 1,
        }
    ]
    assert "injected fadvise failure" in captured.err


def test_runner_rejects_non_empty_evidence_directory(tmp_path):
    stale = tmp_path / "evidence" / "stale.txt"
    stale.parent.mkdir()
    stale.write_text("keep", encoding="utf-8")

    result = subprocess.run(
        [RUNNER_PATH, "a", stale.parent],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "refusing to reuse non-empty evidence directory" in result.stderr
    assert stale.read_text(encoding="utf-8") == "keep"
    assert sorted(stale.parent.iterdir()) == [stale]


def test_runner_owns_cleanup_before_scale_up_and_finalizes_last():
    source = RUNNER_PATH.read_text(encoding="utf-8")
    scale_up = source.index("stamp SCALE_UP_SENT")

    assert source.index("trap cleanup EXIT INT TERM") < scale_up
    cleanup_owned = source.index("CLEANUP_OWNED=1")
    assert cleanup_owned < scale_up
    assert source.rfind("ZERO_CONFIRMED=0", 0, scale_up) > cleanup_owned
    assert source.index("stamp RUNNER_EXIT") < source.index(
        "finalize_evidence || status=1"
    )
    assert source.index("stamp RUN_COMPLETE") > source.index(
        "capture_objects post-teardown"
    )
    assert "SHA256SUMS" not in source[source.index("stamp RUN_COMPLETE") :]
