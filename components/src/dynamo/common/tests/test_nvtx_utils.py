# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.common.utils.nvtx_utils color handling with DYN_NVTX=1.

The nvtx package knows only a few color names and imports matplotlib for any
other. It resolves colors only while a profiler is attached, so an unsupported
name fails exactly when someone profiles (for example at import, through a
decorator). CI runs without a profiler, so these tests check the colors that
nvtx_utils hands to nvtx instead of waiting for nvtx to raise.
"""

import importlib
import re
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

nvtx_colors = pytest.importorskip("nvtx.colors")

DYNAMO_SRC = Path(__file__).resolve().parents[2]
COLOR_LITERAL = re.compile(
    r'_nvtx\.(?:annotate|start_range|range_decorator)\([^)]*color="([a-z_]+)"'
)
NVTX_BUILTIN_NAMES = [
    "green",
    "blue",
    "yellow",
    "purple",
    "rapids",
    "cyan",
    "red",
    "white",
    "darkgreen",
    "orange",
]


@pytest.fixture
def enabled_nvtx_utils(monkeypatch):
    """nvtx_utils reloaded with DYN_NVTX=1 and no matplotlib, as in runtime images."""
    from dynamo.common.utils import nvtx_utils

    try:
        with monkeypatch.context() as patch:
            patch.setenv("DYN_NVTX", "1")
            patch.setitem(sys.modules, "matplotlib", None)
            module = importlib.reload(nvtx_utils)
            assert module.ENABLED
            yield module
    finally:
        importlib.reload(nvtx_utils)


def test_every_color_used_in_dynamo_resolves_without_matplotlib(enabled_nvtx_utils):
    used = {
        match.group(1)
        for path in DYNAMO_SRC.rglob("*.py")
        for match in COLOR_LITERAL.finditer(path.read_text(encoding="utf-8"))
    }
    assert used, "expected NVTX color literals in the Dynamo source tree"
    for name in sorted(used):
        resolved = enabled_nvtx_utils._resolve_color(name)
        assert isinstance(resolved, int), f"{name!r} would need matplotlib"


@pytest.mark.parametrize("name", NVTX_BUILTIN_NAMES)
def test_builtin_names_keep_nvtx_values(enabled_nvtx_utils, name):
    assert enabled_nvtx_utils._resolve_color(name) == nvtx_colors.color_to_hex(name)


@pytest.mark.parametrize("color", [None, 0x80402010])
def test_default_and_integer_colors_pass_through(enabled_nvtx_utils, color):
    assert enabled_nvtx_utils._resolve_color(color) == color


def test_unknown_name_falls_back_to_the_default_color(enabled_nvtx_utils):
    assert enabled_nvtx_utils._resolve_color("not-a-color") is None


def test_invalid_color_with_matplotlib_falls_back_to_the_default_color(
    enabled_nvtx_utils, monkeypatch
):
    def invalid_color(color):
        raise ValueError(f"Invalid RGBA argument: {color!r}")

    monkeypatch.setattr(enabled_nvtx_utils, "color_to_hex", invalid_color)

    assert enabled_nvtx_utils._resolve_color("not-a-color") is None


def test_annotate_passes_a_resolved_color_in_the_dynamo_domain(
    enabled_nvtx_utils, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        enabled_nvtx_utils._nvtx_lib,
        "annotate",
        lambda message=None, **kwargs: calls.append((message, kwargs)),
    )

    enabled_nvtx_utils.annotate("mm:local:send_embeddings", color="magenta")

    assert calls == [
        ("mm:local:send_embeddings", {"color": 0xFF00FF, "domain": "dynamo"})
    ]


def test_manual_ranges_pass_a_resolved_color(enabled_nvtx_utils, monkeypatch):
    class RecordingDomain:
        def __init__(self):
            self.colors = []

        def get_event_attributes(self, message=None, color=None, **_):
            self.colors.append(color)
            return message, color

        def start_range(self, attributes):
            return attributes

        def end_range(self, rng):
            pass

    domain = RecordingDomain()
    monkeypatch.setattr(enabled_nvtx_utils, "_domain", domain)

    rng = enabled_nvtx_utils.start_range("mm:nixl:begin_read", color="pink")
    enabled_nvtx_utils.end_range(rng)

    assert domain.colors == [0xFFC0CB]
