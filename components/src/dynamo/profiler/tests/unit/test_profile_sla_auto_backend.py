# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for auto-backend resolution in the profiler pipeline."""

import pytest

try:
    from dynamo.profiler.rapid import _DEFAULT_NAIVE_BACKEND
except ImportError:
    pytest.skip("dynamo.llm bindings not available", allow_module_level=True)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]


def test_autoscale_sim_resolves_auto_to_default() -> None:
    """_run_autoscale_sim must resolve 'auto' to _DEFAULT_NAIVE_BACKEND before
    constructing TaskConfig, since BackendName('auto') is not a valid enum value.
    """
    import inspect

    from dynamo.profiler.rapid import _run_autoscale_sim

    src = inspect.getsource(_run_autoscale_sim)
    # The function must guard against "auto" before TaskConfig is constructed.
    assert (
        'backend == "auto"' in src
    ), "_run_autoscale_sim must resolve backend='auto' before constructing TaskConfig"
    assert (
        "_DEFAULT_NAIVE_BACKEND" in src
    ), "_run_autoscale_sim must fall back to _DEFAULT_NAIVE_BACKEND when backend='auto'"


def test_autoscale_sim_returns_resolved_backend() -> None:
    """_run_autoscale_sim must include 'resolved_backend' in its result dict so
    profile_sla.py can pass the concrete backend to run_interpolation.
    """
    import inspect

    from dynamo.profiler.rapid import _run_autoscale_sim

    src = inspect.getsource(_run_autoscale_sim)
    assert (
        '"resolved_backend"' in src
    ), "_run_autoscale_sim must return 'resolved_backend' in its result dict"


def test_naive_fallback_resolves_auto_to_default() -> None:
    """_run_naive_fallback must resolve 'auto' to _DEFAULT_NAIVE_BACKEND.

    The naive path is taken when AIC doesn't support the model/system combo;
    it uses build_naive_generator_params and generate_backend_artifacts which
    require a concrete backend string.
    """
    import inspect

    from dynamo.profiler.rapid import _run_naive_fallback

    src = inspect.getsource(_run_naive_fallback)
    assert (
        'backend == "auto"' in src
    ), "_run_naive_fallback must resolve backend='auto' before calling AIC helpers"
    assert "_DEFAULT_NAIVE_BACKEND" in src


def test_default_sim_returns_resolved_backend() -> None:
    """_run_default_sim must include 'resolved_backend' in its result dict.

    When backend='auto', AIC expands to per-backend task configs and the
    winning row carries the concrete backend name; _run_default_sim must
    extract and surface it so run_interpolation never receives 'auto'.
    """
    import inspect

    from dynamo.profiler.rapid import _run_default_sim

    src = inspect.getsource(_run_default_sim)
    assert (
        '"resolved_backend"' in src
    ), "_run_default_sim must return 'resolved_backend' in its result dict"


def test_default_naive_backend_is_concrete() -> None:
    """_DEFAULT_NAIVE_BACKEND must be a concrete backend string, not 'auto'."""
    assert _DEFAULT_NAIVE_BACKEND != "auto"
    assert _DEFAULT_NAIVE_BACKEND in ("vllm", "sglang", "trtllm")
