# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the media-codec allowlist gate (compliance.scan_codecs).

Run from the repo root with the compliance package on the path:

    PYTHONPATH=container python -m pytest container/compliance/tests/test_scan_codecs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from compliance.scan_codecs import CodecPolicy, main, scan_filesystem, scan_sbom

# The real shipped policy — the tests assert against it, not a fixture, so a
# policy edit that would let a non-allowlisted media codec through fails a test here.
_POLICY = CodecPolicy.load(
    Path(__file__).resolve().parents[1] / "policy" / "codec_policy.yaml"
)


def _touch(root: Path, rel: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x7fELF")


def test_in_tree_ffmpeg_is_allowed(tmp_path: Path):
    # Our own /usr/local ffmpeg libs — must classify as allowed, never violate.
    for rel in (
        "usr/local/lib/libavcodec.so.62",
        "usr/local/lib/libswscale.so.9",
        "usr/local/bin/ffmpeg",
    ):
        _touch(tmp_path, rel)
    violations, _exceptions, allowed = scan_filesystem(tmp_path, _POLICY)
    assert violations == []
    assert {a["path"] for a in allowed} == {
        "/usr/local/lib/libavcodec.so.62",
        "/usr/local/lib/libswscale.so.9",
        "/usr/local/bin/ffmpeg",
    }


def test_dali_bundled_libav_is_a_logged_exception(tmp_path: Path):
    _touch(
        tmp_path,
        "usr/local/lib/python3.12/dist-packages/nvidia/dali/.libs/libavcodec-73c99a8b.so.62",
    )
    violations, exceptions, _allowed = scan_filesystem(tmp_path, _POLICY)
    assert violations == []
    assert len(exceptions) == 1
    assert "DALI" in (exceptions[0]["detail"] or "")


@pytest.mark.parametrize(
    "rel",
    [
        "usr/lib/x86_64-linux-gnu/libx264.so.164",
        "usr/lib/x86_64-linux-gnu/libx265.so.199",
        # A third-party bundled libavcodec NOT under /usr/local and NOT DALI.
        "usr/local/lib/python3.12/dist-packages/opencv_python_headless.libs/libavcodec-156beeea.so.62.11.100",
        "opt/venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.1",
    ],
)
def test_nonallowlisted_media_artifacts_are_violations(tmp_path: Path, rel: str):
    _touch(tmp_path, rel)
    violations, _, _ = scan_filesystem(tmp_path, _POLICY)
    assert [v["path"] for v in violations] == ["/" + rel]


def test_unrelated_files_are_ignored(tmp_path: Path):
    for rel in ("usr/lib/libc.so.6", "usr/local/lib/libvpx.so.9", "opt/dynamo/foo.py"):
        _touch(tmp_path, rel)
    violations, exceptions, allowed = scan_filesystem(tmp_path, _POLICY)
    assert (violations, exceptions, allowed) == ([], [], [])


def test_sbom_flags_ffmpeg_below_cve_floor(tmp_path: Path):
    sbom = tmp_path / "s.cdx.json"
    sbom.write_text(
        json.dumps(
            {
                "components": [
                    {"name": "ffmpeg", "version": "8.0.1"},  # < 8.1.2 -> flagged
                    {"name": "ffmpeg", "version": "8.1.2"},  # == floor -> ok
                    {"name": "libvpx", "version": "1.14.1"},  # not denied
                ]
            }
        )
    )
    hits = scan_sbom(sbom, _POLICY)
    assert [h["path"] for h in hits] == ["sbom:ffmpeg@8.0.1"]


def test_fail_on_findings_exit_code(tmp_path: Path):
    _touch(tmp_path, "usr/lib/x86_64-linux-gnu/libx264.so.164")
    assert main(["--root", str(tmp_path), "--fail-on-findings"]) == 1
    # Same tree, report-only: findings reported but exit 0.
    assert main(["--root", str(tmp_path)]) == 0


def test_clean_tree_passes(tmp_path: Path):
    _touch(tmp_path, "usr/local/lib/libavcodec.so.62")  # ours -> allowed
    assert main(["--root", str(tmp_path), "--fail-on-findings"]) == 0
