# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for auto-backend resolution in profile_sla."""

from unittest.mock import patch

import pytest

from dynamo.profiler.profile_sla import _CONCRETE_BACKENDS, _resolve_auto_backend

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.parallel,
]


def test_resolve_auto_backend_returns_first_supported() -> None:
    """_resolve_auto_backend returns the first AIC-supported backend."""
    # trtllm is first in _CONCRETE_BACKENDS; simulate it being supported.
    supported = {"trtllm"}
    with patch(
        "dynamo.profiler.profile_sla.check_model_hardware_support",
        side_effect=lambda model, system, b: b in supported,
    ):
        result = _resolve_auto_backend("Qwen/Qwen3-32B", "b200_sxm")
    assert result == "trtllm"


def test_resolve_auto_backend_skips_unsupported() -> None:
    """_resolve_auto_backend skips unsupported backends and returns the next one."""
    # Only vllm is supported (last in the list).
    supported = {"vllm"}
    with patch(
        "dynamo.profiler.profile_sla.check_model_hardware_support",
        side_effect=lambda model, system, b: b in supported,
    ):
        result = _resolve_auto_backend("Qwen/Qwen3-32B", "b200_sxm")
    assert result == "vllm"


def test_resolve_auto_backend_falls_back_to_vllm_when_none_supported() -> None:
    """_resolve_auto_backend falls back to 'vllm' when no backend is AIC-supported.

    This also covers the aic_supported=False case: backend is still resolved
    unconditionally so that run_interpolation and other downstream consumers
    never receive 'auto'.
    """
    with patch(
        "dynamo.profiler.profile_sla.check_model_hardware_support",
        return_value=False,
    ):
        result = _resolve_auto_backend("unknown-model", "unknown-system")
    assert result == "vllm"


def test_resolve_auto_backend_respects_priority_order() -> None:
    """_resolve_auto_backend honours the _CONCRETE_BACKENDS priority order."""
    # All backends supported — must return the first one in _CONCRETE_BACKENDS.
    with patch(
        "dynamo.profiler.profile_sla.check_model_hardware_support",
        return_value=True,
    ):
        result = _resolve_auto_backend("any-model", "any-system")
    assert result == _CONCRETE_BACKENDS[0]


def test_run_naive_fallback_does_not_override_resolved_backend() -> None:
    """_run_naive_fallback must not silently replace a resolved backend with 'vllm'.

    After profile_sla.py resolves 'auto' to a concrete backend (e.g. 'sglang'),
    that backend must flow into the naive fallback unchanged.  The old guard
    ``if backend == 'auto': backend = 'vllm'`` has been removed; this test
    ensures it does not come back.
    """
    import inspect

    from dynamo.profiler.rapid import _run_naive_fallback

    src = inspect.getsource(_run_naive_fallback)
    assert (
        "auto" not in src or "_DEFAULT_NAIVE_BACKEND" not in src
    ), "_run_naive_fallback must not contain an 'auto' → vllm override guard"
    assert "_DEFAULT_NAIVE_BACKEND" not in src, (
        "_DEFAULT_NAIVE_BACKEND was removed; re-introducing it risks overriding "
        "a legitimately resolved backend like 'sglang' with 'vllm'"
    )
