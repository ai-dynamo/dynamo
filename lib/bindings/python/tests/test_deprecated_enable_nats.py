# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test that the deprecated enable_nats parameter on DistributedRuntime is accepted
and emits a DeprecationWarning.
"""

import asyncio
import inspect
import warnings

import pytest

from dynamo._core import DistributedRuntime

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


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
