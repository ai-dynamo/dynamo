# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the in-tree ffmpeg codec contract in the rendered vLLM Dockerfile.

The contract has two halves that pull against each other, and both must hold at
once:

* The NVDEC decode path needs the ``h264_mp4toannexb`` / ``hevc_mp4toannexb``
  bitstream filters. PyNvVideoCodec resolves them with ``av_bsf_get_by_name()``
  against whichever ``libavcodec.so.62`` the loader mapped first, which is the
  in-tree build; if they are not compiled in, the lookup returns NULL and
  hardware decode fails for every codec.
* No H.264/HEVC/AAC *encoder, decoder or parser* may be built. Those carry a
  codec implementation; the bitstream filters do not, they only reframe an
  already-encoded stream.

Assertions run against the rendered Dockerfile rather than the template so that
a Jinja gate which drops the ffmpeg build out of the vLLM image is caught too.

Run from the repo root with the compliance package on the path:

    PYTHONPATH=container python -m pytest container/compliance/tests/test_ffmpeg_bitstream_filters.py
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pytest
import render
import yaml

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.post_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
]

_CONTAINER = Path(__file__).resolve().parents[2]

# Anything whose implementation is royalty-bearing. Matches ffmpeg's own naming
# for the enable lists (h264, hevc, aac, and the hardware variants).
_DISALLOWED_CODEC_RE = re.compile(r"h\.?26[45]|hevc|aac|nvenc|cuvid|nvdec", re.I)

_REQUIRED_BSFS = ("h264_mp4toannexb", "hevc_mp4toannexb")


def _render_vllm_runtime_dockerfile(out_dir: Path) -> str:
    """Render the vLLM CUDA-13.0 amd64 runtime Dockerfile into ``out_dir``.

    ``render.render()`` writes its output next to the templates it loads, so the
    template tree is copied into ``out_dir`` first to keep the repo checkout
    clean.
    """
    shutil.copytree(_CONTAINER / "templates", out_dir / "templates")
    shutil.copy(_CONTAINER / "Dockerfile.template", out_dir / "Dockerfile.template")

    args = argparse.Namespace(
        framework="vllm",
        device="cuda",
        target="runtime",
        platform="amd64",
        cuda_version="13.0",
        make_efa=False,
        output_short_filename=True,
        show_result=False,
    )
    render.validate_args(args)
    context = yaml.safe_load((_CONTAINER / "context.yaml").read_text())
    render.render(args, context, out_dir)
    return (out_dir / "rendered.Dockerfile").read_text()


def _ffmpeg_configure_flags(dockerfile: str) -> list[str]:
    """Return the flags of the in-tree ffmpeg ``./configure`` invocation.

    Anchored past the libvpx build, which runs its own ``./configure`` earlier in
    the same layer. Comment lines are dropped: BuildKit strips them before the
    shell sees the command, and the surrounding prose mentions the very flag
    names under test.
    """
    anchor = "cd ffmpeg-${FFMPEG_VERSION}"
    assert anchor in dockerfile, "no in-tree ffmpeg build in the rendered Dockerfile"
    body = dockerfile.split(anchor, 1)[1].split("./configure", 1)
    assert len(body) == 2, "no in-tree ffmpeg ./configure in the rendered Dockerfile"
    flags = []
    for raw in body[1].splitlines():
        line = raw.strip()
        if line.startswith("#"):
            continue
        line = line.rstrip("\\").strip()
        for token in line.split():
            if token.startswith("--"):
                flags.append(token)
        # The invocation ends at the first continuation that is not a flag.
        if line and not line.startswith("--"):
            break
    assert flags, "could not parse the ffmpeg ./configure flags"
    return flags


def _enabled(flags: list[str], surface: str) -> set[str]:
    """Collect the comma-separated values of every ``--enable-<surface>=`` flag."""
    prefix = f"--enable-{surface}="
    values: set[str] = set()
    for flag in flags:
        if flag.startswith(prefix):
            values.update(v for v in flag[len(prefix) :].split(",") if v)
    return values


@pytest.fixture(scope="module")
def configure_flags(tmp_path_factory) -> list[str]:
    dockerfile = _render_vllm_runtime_dockerfile(
        tmp_path_factory.mktemp("rendered") / "container"
    )
    return _ffmpeg_configure_flags(dockerfile)


def test_nvdec_reframing_bitstream_filters_are_built(configure_flags):
    # Without these two the NVDEC path's av_bsf_get_by_name() returns NULL and
    # hardware decode fails for every codec. Nothing else pulls them in: no
    # enabled muxer depends on them under --disable-bsfs.
    enabled = _enabled(configure_flags, "bsf")
    missing = [bsf for bsf in _REQUIRED_BSFS if bsf not in enabled]
    assert not missing, (
        f"in-tree ffmpeg would not build {missing}; "
        f"--enable-bsf currently requests {sorted(enabled) or 'nothing'}"
    )


def test_bitstream_filter_baseline_stays_narrow(configure_flags):
    # The two filters above are re-enabled on top of --disable-bsfs, not by
    # dropping the baseline. Losing the baseline would ship ffmpeg's whole
    # bitstream-filter set.
    assert "--disable-bsfs" in configure_flags
    assert "--enable-bsfs" not in configure_flags


@pytest.mark.parametrize("surface", ["encoders", "decoders", "parsers"])
def test_codec_surfaces_are_disabled_by_default(configure_flags, surface):
    assert f"--disable-{surface}" in configure_flags


@pytest.mark.parametrize("surface", ["encoder", "decoder", "parser"])
def test_no_royalty_bearing_codec_is_enabled(configure_flags, surface):
    # The negative control for the change above: fixing NVDEC by re-enabling a
    # software H.264/HEVC decoder instead of a bitstream filter must fail here.
    offenders = sorted(
        name
        for name in _enabled(configure_flags, surface)
        if _DISALLOWED_CODEC_RE.search(name)
    )
    assert not offenders, f"--enable-{surface} names a disallowed codec: {offenders}"
