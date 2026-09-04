# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the SGLang ServerArgs override ladder used by setup_gms."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
from _deps import HAS_GMS

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

from gpu_memory_service.integrations.sglang import _override_server_args

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
]

_OVERRIDES_MODULE = "sglang.srt.arg_groups.overrides"


class ResolvedServerArgs:
    """Double for a resolved SGLang 0.5.18 ServerArgs.

    SGLang 0.5.18 keeps the instance itself read-only once it has been resolved
    and takes launcher-stage changes through ``_late_resolution``. Attribute
    assignment raises, with the message shape a worker sees at startup.
    """

    def __init__(self):
        object.__setattr__(self, "declarations", [])
        object.__setattr__(self, "assignments", [])

    def _late_resolution(self, source, **fields):
        self.declarations.append((source, fields))

    def __setattr__(self, name, value):
        self.assignments.append(name)
        raise AttributeError(
            f"server_args.{name} assigned after resolution; server_args is read-only"
        )


@pytest.fixture
def without_declare_late_resolution(monkeypatch):
    """Hide the module-level declaration API from the ladder.

    The doubles below stand in for SGLang releases that have no
    ``declare_late_resolution``, so the branch has to be unavailable however the
    SGLang in this image is built.
    """
    monkeypatch.setitem(sys.modules, _OVERRIDES_MODULE, ModuleType(_OVERRIDES_MODULE))


def test_resolved_server_args_gets_memory_saver_by_declaration(
    without_declare_late_resolution,
):
    server_args = ResolvedServerArgs()

    _override_server_args(server_args, "dynamo.gms", enable_memory_saver=True)

    assert server_args.declarations == [("dynamo.gms", {"enable_memory_saver": True})]
    assert server_args.assignments == []


def test_legacy_xpu_server_args_gets_memory_saver_by_assignment(
    without_declare_late_resolution,
):
    # SGLang 0.5.11 on the XPU pin has neither override API and stays mutable.
    server_args = SimpleNamespace(enable_memory_saver=False)

    _override_server_args(server_args, "dynamo.gms", enable_memory_saver=True)

    assert server_args.enable_memory_saver is True


def test_installed_sglang_accepts_memory_saver_after_resolution():
    server_args = _installed_sglang_server_args()

    _override_server_args(server_args, "dynamo.gms", enable_memory_saver=True)

    try:
        resolved = _resolved_view(server_args)
    except Exception as exc:
        pytest.skip(
            f"installed SGLang could not resolve ServerArgs ({type(exc).__name__})"
        )
    assert resolved.enable_memory_saver is True


def _installed_sglang_server_args():
    """Build a ServerArgs from the installed SGLang, or skip the test."""
    try:
        from sglang.srt.server_args import ServerArgs
    except Exception as exc:
        pytest.skip(f"SGLang is not importable here ({type(exc).__name__})")

    try:
        return ServerArgs(model_path="Qwen/Qwen3-0.6B")
    except Exception as exc:
        pytest.skip(
            f"installed SGLang could not build ServerArgs ({type(exc).__name__})"
        )


def _resolved_view(server_args):
    """Return SGLang's resolved projection, mirroring _compat.resolved_server_args."""
    resolve = getattr(server_args, "_resolved", None)
    if callable(resolve):
        return resolve()
    try:
        from sglang.srt.arg_groups.overrides import resolved_view
    except ImportError:
        return server_args
    return resolved_view(server_args)
