# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.vllm.strict_rl_controller import StrictRLWeightTransferController

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.timeout(20),
]


@pytest.fixture
def controller():
    return StrictRLWeightTransferController(
        SimpleNamespace(
            pause_generation=AsyncMock(),
            resume_generation=AsyncMock(),
            compute_weight_checksums=AsyncMock(),
            get_weight_version=AsyncMock(return_value="wrong-policy"),
        ),
        128,
        asyncio.Lock(),
        Mock(),
        device_uuids=("GPU-test",),
    )


@pytest.mark.parametrize("exception", [RuntimeError, ValueError])
@pytest.mark.parametrize("already_failed", [False, True])
async def test_native_failure_raises_safe_error_and_fences(
    controller, exception, already_failed, caplog
):
    if already_failed:
        controller.session.fail()
    failure = exception("private native metadata")
    controller.engine_client.pause_generation.side_effect = failure
    with pytest.raises(RuntimeError, match="restart worker") as raised:
        await controller.pause_generation({"mode": "abort"})
    assert raised.value.__cause__ is failure
    assert "private native metadata" not in str(raised.value)
    assert "private native metadata" not in caplog.text
    assert controller.session.snapshot().failed
    assert (await controller.resume_generation({}))["status"] == "error"


async def test_protocol_validation_returns_error_without_fencing(controller):
    assert (await controller.resume_generation({}))["status"] == "error"
    assert (await controller.flush_cache({"unexpected": True}))["status"] == "error"
    assert not controller.session.snapshot().failed


@pytest.mark.parametrize("reply", [[], [{}], [{1: "digest"}], [{"weight": 1}]])
async def test_malformed_native_checksums_raise_and_fence(controller, reply):
    controller.engine_client.compute_weight_checksums.return_value = reply
    with pytest.raises(RuntimeError, match="restart worker"):
        await controller.get_weight_checksums({})
    assert controller.session.snapshot().failed


async def test_native_version_mismatch_raises_and_fences(controller, advance_session):
    await advance_session(controller.session)
    with pytest.raises(RuntimeError, match="restart worker"):
        await controller.get_weight_version({})
    assert controller.session.snapshot().failed


async def test_failed_cache_reset_raises_and_fences(controller, advance_session):
    await advance_session(controller.session, "finish")
    controller.engine_client.reset_prefix_cache = AsyncMock(return_value=False)
    with pytest.raises(RuntimeError, match="restart worker"):
        await controller.flush_cache({})
    assert controller.session.snapshot().failed
