# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402 - optional vLLM dependency is checked before Dynamo imports

from dataclasses import replace

import pytest

pytest.importorskip("vllm.lora.request")

from dynamo.common.lora.runtime import (
    ResolvedLoRA,
    RuntimeLoRANotFoundError,
    RuntimeLoRAPluginError,
    RuntimeLoRAResolverUnavailableError,
)
from dynamo.llm import HttpError
from dynamo.vllm.runtime_lora_cache import cache_reservation
from dynamo.vllm.tests.test_vllm_runtime_lora import _coordinator, _request, _snapshot

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.parametrize(
    "failure,status",
    [
        (None, None),
        (RuntimeLoRANotFoundError, 404),
        (RuntimeLoRAPluginError, 500),
        (RuntimeLoRAResolverUnavailableError, 503),
    ],
)
@pytest.mark.asyncio
async def test_zero_download_budget_allows_cached_snapshot_without_masking_provider_errors(
    monkeypatch, tmp_path, failure, status
):
    snapshot = _snapshot(tmp_path)
    snapshot_bytes = sum(path.stat().st_size for path in snapshot.iterdir())
    budgets = []

    async def resolve(**kwargs):
        budget = kwargs["context"].max_download_bytes
        budgets.append(budget)
        assert budget == 0
        if failure is not None:
            raise failure("provider-private-error")
        return ResolvedLoRA(snapshot, "digest:immutable")

    handler, coordinator = _coordinator(monkeypatch, tmp_path, resolve)
    coordinator._settings = replace(
        coordinator.settings,
        max_download_bytes=snapshot_bytes + 64,
        max_cache_bytes=snapshot_bytes + 64,
    )
    async with cache_reservation(
        handler._lora_state, tmp_path, coordinator.settings
    ) as reservation:
        assert reservation == 64
        (tmp_path / "other-download.partial").write_bytes(bytes(1))
        request = _request("wandb-artifact:///entity/project/cached:v1")
        if failure is None:
            result = await coordinator.ensure_from_request(request, "cached")
            assert result.lora_path == str(snapshot)
            handler.engine_client.add_lora.assert_awaited_once()
        else:
            with pytest.raises(HttpError) as error:
                await coordinator.ensure_from_request(request, "missing")
            assert error.value.code == status
            assert "provider-private-error" not in str(error.value)
            handler.engine_client.add_lora.assert_not_awaited()
        assert handler._lora_state.runtime_cache_reserved_bytes == 64
    assert budgets == [0]
    assert handler._lora_state.runtime_cache_reserved_bytes == 0
    await coordinator.close()
