# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the SGLang ServerArgs override ladder used by setup_gms."""

from types import SimpleNamespace

import pytest
from _deps import HAS_GMS

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

from gpu_memory_service.integrations import sglang as gms_sglang
from gpu_memory_service.integrations.sglang import override_server_args

try:
    from sglang.srt.arg_groups.overrides import resolved_view
except ImportError:
    # SGLang #36255 exposes ServerArgs._resolved() instead.
    resolved_view = None

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
]


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


class LegacyOverrideServerArgs:
    """Double for SGLang 0.5.17 ServerArgs, which takes changes through override()."""

    def __init__(self):
        self.overrides = []

    def override(self, source, **fields):
        self.overrides.append((source, fields))


@pytest.fixture
def without_declare_late_resolution(monkeypatch):
    """Hide the module-level declaration API from the ladder.

    The doubles below stand in for SGLang releases that have no
    ``declare_late_resolution``, so the branch has to be unavailable however the
    SGLang in this image is built.
    """
    monkeypatch.setattr(gms_sglang, "declare_late_resolution", None)


def test_declaration_api_receives_source_and_fields(monkeypatch):
    # Current SGLang exposes the declaration as a module-level function, which
    # takes the ServerArgs as its first argument.
    calls = []
    monkeypatch.setattr(
        gms_sglang,
        "declare_late_resolution",
        lambda server_args, source, **fields: calls.append((source, fields)),
    )
    server_args = ResolvedServerArgs()

    override_server_args(server_args, "dynamo.gms", enable_memory_saver=True)

    assert calls == [("dynamo.gms", {"enable_memory_saver": True})]
    # The ladder stops at the first available branch.
    assert server_args.declarations == []
    assert server_args.assignments == []


def test_resolved_server_args_gets_memory_saver_by_declaration(
    without_declare_late_resolution,
):
    server_args = ResolvedServerArgs()

    override_server_args(server_args, "dynamo.gms", enable_memory_saver=True)

    assert server_args.declarations == [("dynamo.gms", {"enable_memory_saver": True})]
    assert server_args.assignments == []


def test_legacy_override_server_args_gets_memory_saver_by_override(
    without_declare_late_resolution,
):
    server_args = LegacyOverrideServerArgs()

    override_server_args(server_args, "dynamo.gms", enable_memory_saver=True)

    assert server_args.overrides == [("dynamo.gms", {"enable_memory_saver": True})]
    # The field goes through override() rather than landing as an attribute.
    assert not hasattr(server_args, "enable_memory_saver")


def test_legacy_xpu_server_args_gets_memory_saver_by_assignment(
    without_declare_late_resolution,
):
    # SGLang 0.5.11 on the XPU pin has neither override API and stays mutable.
    server_args = SimpleNamespace(enable_memory_saver=False)

    override_server_args(server_args, "dynamo.gms", enable_memory_saver=True)

    assert server_args.enable_memory_saver is True


def test_installed_sglang_accepts_memory_saver_after_resolution():
    server_args = _installed_sglang_server_args()

    override_server_args(server_args, "dynamo.gms", enable_memory_saver=True)

    assert _resolved_view(server_args).enable_memory_saver is True


def _installed_sglang_server_args():
    """Build a ServerArgs from the installed SGLang.

    SGLang is an optional dependency of ``gpu-memory-service``, so an image
    without it skips. Anything else -- a ServerArgs that will not build, a
    resolution that drops the declaration -- is the incompatibility this test
    exists to catch and must fail rather than skip.
    """
    server_args_module = pytest.importorskip(
        "sglang.srt.server_args",
        reason="SGLang is not installed in this test image",
    )
    return server_args_module.ServerArgs(model_path="Qwen/Qwen3-0.6B")


def _resolved_view(server_args):
    """Return SGLang's resolved projection, mirroring _compat.resolved_server_args."""
    resolve = getattr(server_args, "_resolved", None)
    if callable(resolve):
        return resolve()
    if resolved_view is None:
        return server_args
    return resolved_view(server_args)
