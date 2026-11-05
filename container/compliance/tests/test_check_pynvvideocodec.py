# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the PyNvVideoCodec install guard (compliance.check_pynvvideocodec).

Run from the repo root:

    PYTHONPATH=container python -m pytest container/compliance/tests/test_check_pynvvideocodec.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from compliance.check_pynvvideocodec import DEFAULT_POLICY, GuardError, check, main
from compliance.scan_codecs import CodecPolicy

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.post_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
]

PINNED = "2.2.3"
# The shipped policy, so a policy edit that changes a verdict fails here.
POLICY = CodecPolicy.load(DEFAULT_POLICY)
GOOD_LIBS = ("libavformat.so.61", "libavutil.so.59")


def _install(
    root: Path,
    *,
    libs=GOOD_LIBS,
    tarballs=("ffmpeg-7.1.tar.xz",),
    declared=None,
    version=PINNED,
    name="PyNvVideoCodec",
) -> Path:
    """Lay out one wheel under ``root`` the way pip does; return its site dir."""
    site = root / "usr/local/lib/python3.12/dist-packages"
    external = root / "usr/local/external/ffmpeg"
    dist = site / f"{name}-{version}.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    )
    pkg = site / "PyNvVideoCodec"
    pkg.mkdir(exist_ok=True)
    for lib in libs:
        (pkg / lib).parent.mkdir(parents=True, exist_ok=True)
        (pkg / lib).write_bytes(b"\x7fELF")
    external.mkdir(parents=True, exist_ok=True)
    for tarball in tarballs:
        (external / tarball).write_bytes(b"")
    rows = declared if declared is not None else tarballs
    (dist / "RECORD").write_text(
        "".join(f"{os.path.relpath(external / t, site)},,\n" for t in rows)
    )
    return site


def _check(*sites: Path, policy: CodecPolicy = POLICY) -> None:
    check(PINNED, policy, [str(s) for s in sites])


def test_a_clean_install_passes(tmp_path, capsys):
    _check(_install(tmp_path))
    assert "libavformat.so.61" in capsys.readouterr().out


def test_a_version_other_than_the_pin_fails(tmp_path):
    with pytest.raises(GuardError, match="pins 2.2.3"):
        _check(_install(tmp_path, version="2.2.2"))


def test_a_second_distribution_fails(tmp_path):
    # The base image's copy surviving beside the new one.
    first = _install(tmp_path / "a")
    second = _install(tmp_path / "b", version="2.1.0", name="pynvvideocodec")
    with pytest.raises(GuardError, match="exactly one"):
        _check(first, second)


def test_no_distribution_fails(tmp_path):
    with pytest.raises(GuardError, match="exactly one"):
        _check(tmp_path)


def test_a_second_ffmpeg_source_tarball_fails(tmp_path):
    site = _install(
        tmp_path,
        tarballs=("ffmpeg-7.1.tar.xz", "ffmpeg-6.0.tar.xz"),
        declared=("ffmpeg-7.1.tar.xz",),
    )
    with pytest.raises(GuardError, match="exactly one bundled FFmpeg source tarball"):
        _check(site)


def test_a_record_without_a_source_tarball_fails(tmp_path):
    with pytest.raises(GuardError, match="source tarball in the RECORD"):
        _check(_install(tmp_path, declared=()))


@pytest.mark.parametrize(
    "lib",
    [
        "libavcodec.so.61",
        "libavcodec-1a2b3c4d.so.61",
        "libavdevice.so.61",
        "libavfilter.so.10",
        "libswscale.so.8",
        "libswresample.so.5",
        "libpostproc.so.58",
        "libx264.so.164",
        "libx265.so.209",
        "libfdk-aac.so.2",
        ".libs/libx264-1a2b3c4d.so.164",
        "libavcodec_x.so.61",
        "libswscale_x.so.8",
    ],
)
def test_a_denied_library_fails(tmp_path, lib):
    with pytest.raises(GuardError, match="denies"):
        _check(_install(tmp_path, libs=(*GOOD_LIBS, lib)))


@pytest.mark.parametrize("libs", [(), ("libavformat.so.61",), ("libavutil.so.59",)])
def test_an_empty_or_partial_package_fails(tmp_path, libs):
    # Each required library must be bundled, or the negative checks pass vacuously.
    with pytest.raises(GuardError, match="bundles no"):
        _check(_install(tmp_path, libs=libs))


def test_the_waiver_is_scoped_to_the_package_directory(tmp_path):
    # The policy waives libavutil beside the package, not a copy grafted elsewhere.
    with pytest.raises(GuardError, match="denies"):
        _check(_install(tmp_path, libs=(*GOOD_LIBS, ".libs/libavutil-1a2b3c4d.so.59")))


def test_a_family_added_to_the_policy_reaches_the_guard(tmp_path):
    site = _install(tmp_path, libs=(*GOOD_LIBS, "libfoo.so.1"))
    _check(site)
    doc = yaml.safe_load(DEFAULT_POLICY.read_text(encoding="utf-8"))
    doc["deny_globs"].append("**/libfoo.so*")
    edited = tmp_path / "policy.yaml"
    edited.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(GuardError, match="libfoo"):
        _check(site, policy=CodecPolicy.load(edited))


def test_main_exit_codes(tmp_path, monkeypatch, capsys):
    site = _install(tmp_path)
    monkeypatch.setattr("sys.path", [str(site)])
    assert main(["--pinned", PINNED]) == 0
    assert main(["--pinned", "9.9.9"]) == 1
    assert "ERROR:" in capsys.readouterr().err
