# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test that deprecated parameters are still accepted and emit DeprecationWarnings:

- DistributedRuntime(enable_nats=...)
- @dynamo_worker(enable_nats=...)
- create_runtime(use_kv_events=...)

Downstream components may still pass these kwargs. We must keep accepting
them for N+1 backwards compatibility while steering callers toward the
new auto-detection behaviour.
"""

import asyncio
import inspect
import warnings

import pytest

from dynamo._core import DistributedRuntime
from dynamo.common.utils.runtime import create_runtime
from dynamo.runtime import dynamo_worker

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


# ---------------------------------------------------------------------------
# DistributedRuntime tests
# ---------------------------------------------------------------------------


def test_enable_nats_parameter_in_signature():
    """DistributedRuntime.__init__ should still accept enable_nats as an optional kwarg."""
    sig = inspect.signature(DistributedRuntime)
    assert "enable_nats" in sig.parameters
    param = sig.parameters["enable_nats"]
    assert param.default is None


@pytest.mark.forked
@pytest.mark.asyncio
async def test_enable_nats_emits_deprecation_warning(discovery_backend, request_plane):
    """Passing enable_nats should emit a DeprecationWarning but otherwise work."""
    loop = asyncio.get_running_loop()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runtime = DistributedRuntime(
            loop, discovery_backend, request_plane, enable_nats=True
        )
    deprecation_warnings = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert len(deprecation_warnings) == 1
    assert "enable_nats" in str(deprecation_warnings[0].message)
    runtime.shutdown()


@pytest.mark.forked
@pytest.mark.asyncio
async def test_no_warning_without_enable_nats(discovery_backend, request_plane):
    """Omitting enable_nats should not emit a DeprecationWarning."""
    loop = asyncio.get_running_loop()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runtime = DistributedRuntime(loop, discovery_backend, request_plane)
    deprecation_warnings = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert len(deprecation_warnings) == 0
    runtime.shutdown()


# ---------------------------------------------------------------------------
# dynamo_worker() decorator tests
# ---------------------------------------------------------------------------


def test_dynamo_worker_accepts_enable_nats_kwarg():
    """dynamo_worker() should accept enable_nats as an optional keyword argument."""
    sig = inspect.signature(dynamo_worker)
    assert "enable_nats" in sig.parameters
    param = sig.parameters["enable_nats"]
    assert param.default is None


def test_dynamo_worker_enable_nats_true_emits_warning():
    """@dynamo_worker(enable_nats=True) should emit a DeprecationWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        decorator = dynamo_worker(enable_nats=True)
    deprecation_warnings = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert len(deprecation_warnings) == 1
    assert "enable_nats" in str(deprecation_warnings[0].message)
    assert callable(decorator)


def test_dynamo_worker_enable_nats_false_emits_warning():
    """@dynamo_worker(enable_nats=False) should also emit a DeprecationWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        decorator = dynamo_worker(enable_nats=False)
    deprecation_warnings = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert len(deprecation_warnings) == 1
    assert "enable_nats" in str(deprecation_warnings[0].message)
    assert callable(decorator)


def test_dynamo_worker_no_args_no_warning():
    """@dynamo_worker() without enable_nats should not emit a DeprecationWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        decorator = dynamo_worker()
    deprecation_warnings = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert len(deprecation_warnings) == 0
    assert callable(decorator)


def test_dynamo_worker_returns_working_decorator():
    """The decorator returned by dynamo_worker(enable_nats=True) should wrap a function."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")

        @dynamo_worker(enable_nats=True)
        async def _sample_worker(runtime, *args, **kwargs):
            pass

    assert asyncio.iscoroutinefunction(_sample_worker)


# ---------------------------------------------------------------------------
# create_runtime() tests (use_kv_events deprecation)
# ---------------------------------------------------------------------------


def test_create_runtime_accepts_use_kv_events_kwarg():
    """create_runtime() should accept use_kv_events as an optional keyword argument."""
    sig = inspect.signature(create_runtime)
    assert "use_kv_events" in sig.parameters
    param = sig.parameters["use_kv_events"]
    assert param.default is None


@pytest.mark.forked
@pytest.mark.asyncio
async def test_create_runtime_use_kv_events_true_emits_warning(
    discovery_backend, request_plane
):
    """create_runtime(use_kv_events=True) should emit a DeprecationWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runtime, loop = create_runtime(
            discovery_backend=discovery_backend,
            request_plane=request_plane,
            event_plane="nats",
            use_kv_events=True,
        )
    deprecation_warnings = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert len(deprecation_warnings) == 1
    assert "use_kv_events" in str(deprecation_warnings[0].message)
    runtime.shutdown()


@pytest.mark.forked
@pytest.mark.asyncio
async def test_create_runtime_use_kv_events_false_emits_warning(
    discovery_backend, request_plane
):
    """create_runtime(use_kv_events=False) should also emit a DeprecationWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runtime, loop = create_runtime(
            discovery_backend=discovery_backend,
            request_plane=request_plane,
            event_plane="nats",
            use_kv_events=False,
        )
    deprecation_warnings = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert len(deprecation_warnings) == 1
    assert "use_kv_events" in str(deprecation_warnings[0].message)
    runtime.shutdown()


@pytest.mark.forked
@pytest.mark.asyncio
async def test_create_runtime_no_use_kv_events_no_warning(
    discovery_backend, request_plane
):
    """Omitting use_kv_events should not emit a DeprecationWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runtime, loop = create_runtime(
            discovery_backend=discovery_backend,
            request_plane=request_plane,
            event_plane="nats",
        )
    deprecation_warnings = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert len(deprecation_warnings) == 0
    runtime.shutdown()
