# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assert the shipped runtime images carry no software H.264/H.265/AAC decoder.

The in-tree FFmpeg is already guarded where it is built: `wheel_builder.Dockerfile`
enumerates `-encoders -decoders -parsers` and fails the build on a match. That
guard runs in the builder stage, so it proves what was *produced*, not what the
runtime image finally *ships* — a base-image bump, a stray copy, or an ffmpeg
earlier on `PATH` would all slip past it.

This closes that gap from the application's point of view: resolve FFmpeg the way
Dynamo's encode path does, and assert the formats we deliberately do not carry in
software are absent. NVIDIA DALI took the same approach when it trimmed its own
FFmpeg build (NVIDIA/DALI#6352): rather than skipping tests for codecs it no
longer supports, it asserts the failure, so a silent reintroduction is caught.

Every negative assertion here is paired with a positive one. A missing or broken
FFmpeg would otherwise satisfy "no H.264 decoder" trivially and the test would
pass while proving nothing.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess

import pytest

# Deliberately NO framework marker here. tests/conftest.py skips an item when any
# framework marker it carries names a module that is absent, so listing all three
# would mean "requires vllm AND sglang AND tensorrt_llm" -- true in no image, and
# the whole module would skip everywhere while looking present. Each image is
# covered by its own single-marker entry point at the bottom of this file.
pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.post_merge,
    pytest.mark.gpu_0,
]

# Formats intentionally absent from the software stack. Hardware decode covers
# H.264/H.265 via NVDEC; audio has no hardware path and is opt-in only.
# Mirrors the build-time guard's pattern in wheel_builder.Dockerfile: `aac` is
# anchored to a word start so it cannot match inside an unrelated identifier.
_DISALLOWED_RE = re.compile(r"h\.?264|h\.?265|hevc|(?:^|\s)aac|nvenc|cuvid|nvdec", re.I)
# Present by construction, so a broken FFmpeg cannot masquerade as a pass.
_REQUIRED = ("vp9",)

_SURFACES = ("encoders", "decoders", "parsers")


def _ffmpeg() -> str:
    """Resolve FFmpeg the way the encode path does."""
    exe = os.environ.get("IMAGEIO_FFMPEG_EXE")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _surface(exe: str, surface: str) -> str:
    proc = subprocess.run(
        [exe, "-hide_banner", f"-{surface}"], capture_output=True, text=True
    )
    return proc.stdout


def _surfaces() -> dict[str, str]:
    exe = _ffmpeg()
    try:
        out = {s: _surface(exe, s) for s in _SURFACES}
    except (OSError, FileNotFoundError) as exc:
        pytest.fail(f"could not run the shipped ffmpeg ({exe}): {exc}")
    if not any(out.values()):
        pytest.fail(f"ffmpeg at {exe} produced no codec listing; cannot verify")
    return out


def _assert_required_codecs_present(surfaces: dict[str, str]) -> None:
    """Sanity: the listings are real, so the absence checks mean something."""
    listing = "\n".join(surfaces.values()).lower()
    for name in _REQUIRED:
        assert name in listing, (
            f"{name} missing from the shipped ffmpeg -- the build is broken, and "
            "the absence assertions here would pass vacuously"
        )


def _assert_no_software_codecs(surfaces: dict[str, str]) -> None:
    """No implementation-carrying surface may expose the excluded formats.

    Checks encoders/decoders/parsers rather than ``-codecs``, which lists names
    even when nothing is built. Bitstream filters are deliberately not checked:
    they reframe an already-encoded stream, carry no codec implementation, and
    ``h264_mp4toannexb`` is needed to feed hardware decode. This mirrors the
    build-time guard in wheel_builder.Dockerfile, including its pattern -- `aac`
    is anchored so it cannot match inside an unrelated word.
    """
    for surface in _SURFACES:
        listing = surfaces[surface]
        hits = [ln for ln in listing.splitlines() if _DISALLOWED_RE.search(ln)]
        assert not hits, (
            f"shipped ffmpeg exposes an excluded format via -{surface}; the "
            "runtime image carries a software implementation it should not:\n"
            + "\n".join(hits)
        )


def _assert_python_carriers_absent() -> None:
    """The Python decode carriers are absent from an unmodified image.

    They are installable at runtime behind an opt-in switch, and tests that do so
    are marked ``installs_extra_dependencies``. An unmodified image must not have
    them, or a green multimodal run says nothing about what customers receive.
    """
    for module, package in (("cv2", "opencv-python"), ("av", "PyAV")):
        spec = importlib.util.find_spec(module)
        assert spec is None, (
            f"{package} is installed ({module} at {spec.origin}); an unmodified "
            "runtime image is expected to ship without it"
        )


def _check_image() -> None:
    surfaces = _surfaces()
    _assert_required_codecs_present(surfaces)
    _assert_no_software_codecs(surfaces)
    _assert_python_carriers_absent()


# One entry point per image. Each carries a single framework marker so it runs in
# that backend's lane and is skipped in the others -- tests/conftest.py skips an
# item whose framework marker names an absent module, so a single module marked
# with all three would skip everywhere and silently prove nothing.


@pytest.mark.vllm
def test_vllm_image_ships_no_software_video_codecs() -> None:
    _check_image()


@pytest.mark.sglang
def test_sglang_image_ships_no_software_video_codecs() -> None:
    _check_image()


@pytest.mark.trtllm
def test_trtllm_image_ships_no_software_video_codecs() -> None:
    _check_image()
