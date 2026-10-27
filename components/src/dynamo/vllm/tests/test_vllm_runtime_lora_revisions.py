# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402 - optional vLLM dependency is checked before Dynamo imports

from unittest.mock import AsyncMock

import pytest

pytest.importorskip("vllm.lora.request")

from dynamo.common.lora.runtime import ResolvedLoRA
from dynamo.llm import HttpError
from dynamo.vllm import runtime_lora as runtime_mod
from dynamo.vllm.tests.test_vllm_runtime_lora import _coordinator, _request, _snapshot

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.parametrize("changed_field", ["revision", "path", None])
@pytest.mark.asyncio
async def test_evicted_source_keeps_worker_lifetime_snapshot_identity(
    monkeypatch, tmp_path, changed_field
):
    monkeypatch.setenv("DYN_LORA_MAX_RESIDENT_RUNTIME_LORAS", "1")
    snapshot = _snapshot(tmp_path)
    replacement_root = tmp_path / "replacement"
    replacement_root.mkdir()
    replacement = _snapshot(replacement_root)
    original = ResolvedLoRA(snapshot, "digest:original")
    reloaded = ResolvedLoRA(
        replacement if changed_field == "path" else snapshot,
        "digest:changed" if changed_field == "revision" else "digest:original",
    )
    resolver = AsyncMock(side_effect=[original, original, reloaded])
    handler, coordinator = _coordinator(monkeypatch, tmp_path, resolver)
    source = "wandb-artifact:///entity/project/first:v1"
    first = await coordinator.ensure_from_request(_request(source), "first")
    await coordinator.release_pending_admission("first")
    second = await coordinator.ensure_from_request(
        _request("wandb-artifact:///entity/project/second:v1"), "second"
    )
    await coordinator.release_pending_admission("second")

    if changed_field is None:
        result = await coordinator.ensure_from_request(_request(source), "reload")
        assert result.lora_name == first.lora_name
        assert result.lora_path == first.lora_path
        assert handler.engine_client.add_lora.await_count == 3
    else:
        with pytest.raises(HttpError, match="runtime_lora_source_changed") as error:
            await coordinator.ensure_from_request(_request(source), "reload")
        assert error.value.code == 500
        assert set(handler._lora_state.runtime_loras) == {second.lora_name}
        assert handler.engine_client.add_lora.await_count == 2
        assert handler.engine_client.remove_lora.await_count == 1

    await coordinator.close()


@pytest.mark.asyncio
async def test_worker_identity_history_is_bounded_without_forgetting_evicted_keys(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(runtime_mod, "_MAX_RUNTIME_IDENTITIES", 2, raising=False)
    monkeypatch.setenv("DYN_LORA_MAX_RESIDENT_RUNTIME_LORAS", "1")
    snapshot = _snapshot(tmp_path)
    resolutions = []

    async def resolve(**kwargs):
        resolutions.append(kwargs["source_uri"])
        return ResolvedLoRA(snapshot, "digest:immutable")

    _handler, coordinator = _coordinator(monkeypatch, tmp_path, resolve)
    first_source = "wandb-artifact:///entity/project/first:v1"
    for index, source in enumerate(
        [first_source, "wandb-artifact:///entity/project/second:v1"]
    ):
        await coordinator.ensure_from_request(_request(source), str(index))
        await coordinator.release_pending_admission(str(index))
    with pytest.raises(HttpError, match="lora_capacity_exceeded"):
        await coordinator.ensure_from_request(
            _request("wandb-artifact:///entity/project/third:v1"), "third"
        )
    assert len(coordinator._snapshot_identities) == 2
    assert len(resolutions) == 2
    result = await coordinator.ensure_from_request(_request(first_source), "reload")
    assert result.lora_name == _request(first_source)["routing"]["lora_name"]
    assert len(coordinator._snapshot_identities) == 2
    assert len(resolutions) == 3
    await coordinator.close()
