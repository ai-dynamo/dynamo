# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import dynamo.sglang.capacity as capacity_module
from dynamo.sglang.capacity import (
    get_hicache_native_offloading_capacity,
    get_spec_decode_runtime_data,
    resolve_engine_max_num_seqs,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.parametrize(
    ("states", "max_running_requests", "expected"),
    [
        ([7, 6, 5], 999, 15),
        ([7, True, 5], 21, 21),
    ],
)
@pytest.mark.asyncio
async def test_engine_sequence_capacity_requires_complete_effective_local_ranks(
    states, max_running_requests, expected
):
    engine = SimpleNamespace(
        tokenizer_manager=SimpleNamespace(
            get_internal_state=AsyncMock(
                return_value=[
                    {"effective_max_running_requests_per_dp": value} for value in states
                ]
            )
        )
    )

    capacity = await resolve_engine_max_num_seqs(
        engine,
        SimpleNamespace(max_running_requests=max_running_requests, dp_size=3),
        local_dp_size=3,
    )

    assert capacity == expected


@pytest.mark.parametrize("failure", ["error", "timeout"])
@pytest.mark.asyncio
async def test_engine_sequence_capacity_falls_back_to_configured_per_rank(
    failure, monkeypatch, caplog
):
    release = asyncio.Event()

    async def get_internal_state():
        if failure == "error":
            raise RuntimeError("not supported")
        await release.wait()
        return []

    monkeypatch.setattr(capacity_module, "_INTERNAL_STATE_TIMEOUT_SECONDS", 0.01)
    engine = SimpleNamespace(
        tokenizer_manager=SimpleNamespace(get_internal_state=get_internal_state)
    )

    capacity = await resolve_engine_max_num_seqs(
        engine,
        SimpleNamespace(max_running_requests=24, dp_size=4),
        local_dp_size=2,
    )

    assert capacity == 12
    assert "falling back to configured max_running_requests" in caplog.text
    if failure == "timeout":
        release.set()
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_internal_state_timeout_does_not_block_later_calls(monkeypatch):
    # Keep this backend-heavy import lazy so lightweight collection does not
    # require an installed SGLang package.
    from sglang.srt.managers.communicator import FanOutCommunicator

    sent_requests = []
    communicator = FanOutCommunicator(sent_requests.append, fan_out=1)
    engine = SimpleNamespace(
        tokenizer_manager=SimpleNamespace(
            get_internal_state=lambda: communicator(object())
        )
    )
    server_args = SimpleNamespace(max_running_requests=8, dp_size=1)

    monkeypatch.setattr(capacity_module, "_INTERNAL_STATE_TIMEOUT_SECONDS", 0.01)
    assert await resolve_engine_max_num_seqs(engine, server_args, 1) == 8

    monkeypatch.setattr(capacity_module, "_INTERNAL_STATE_TIMEOUT_SECONDS", 1.0)
    second_capacity = asyncio.create_task(
        resolve_engine_max_num_seqs(engine, server_args, 1)
    )
    for _ in range(100):
        if len(communicator._ready_queue) == 1:
            break
        await asyncio.sleep(0)
    assert len(communicator._ready_queue) == 1

    communicator.handle_recv({"effective_max_running_requests_per_dp": 5})
    for _ in range(100):
        if len(sent_requests) == 2:
            break
        await asyncio.sleep(0)
    assert len(sent_requests) == 2

    communicator.handle_recv({"effective_max_running_requests_per_dp": 4})

    assert await asyncio.wait_for(second_capacity, timeout=1.0) == 4
    await asyncio.sleep(0)
    assert not capacity_module._IN_FLIGHT_INTERNAL_STATE_TASKS


@pytest.mark.asyncio
async def test_engine_sequence_capacity_returns_none_without_usable_limit():
    engine = SimpleNamespace(
        tokenizer_manager=SimpleNamespace(
            get_internal_state=AsyncMock(return_value={"other": 1})
        )
    )

    capacity = await resolve_engine_max_num_seqs(
        engine,
        SimpleNamespace(max_running_requests=None, dp_size=1),
        local_dp_size=1,
    )

    assert capacity is None


@pytest.mark.asyncio
async def test_registration_reports_resolved_local_engine_capacity(monkeypatch):
    # Keep this backend-heavy import lazy: importing it at module scope breaks
    # the sglang-light `pytest-marker-report` collection environment.
    from dynamo.sglang import register

    engine = SimpleNamespace(
        _scheduler_init_result=SimpleNamespace(scheduler_infos=[{}]),
    )
    server_args = SimpleNamespace(
        context_length=4096,
        disaggregation_mode=None,
        max_prefill_tokens=None,
        page_size=16,
        speculative_algorithm="NONE",
    )
    dynamo_args = register.DynamoConfig()
    dynamo_args.enable_local_indexer = False
    resolver = AsyncMock(return_value=15)
    capacity = SimpleNamespace(
        max_num_seqs=None,
        max_num_batched_tokens=None,
        total_kv_blocks=None,
    )

    monkeypatch.setattr(register, "resolve_engine_max_num_seqs", resolver)
    monkeypatch.setattr(register, "model_card_dp_rank_bounds", lambda _: (2, 5))
    monkeypatch.setattr(register, "get_sglang_worker_group_id", lambda _: None)
    monkeypatch.setattr(
        register, "_get_bootstrap_info_for_config", lambda _: (None, None)
    )
    monkeypatch.setattr(register, "_get_mooncake_runtime_data", lambda _: None)
    monkeypatch.setattr(register, "runtime_capacity", lambda *_: capacity)

    runtime_config = await register._get_runtime_config(
        engine, server_args, dynamo_args
    )

    resolver.assert_awaited_once_with(engine, server_args, 3)
    assert runtime_config.engine_max_num_seqs == 15


@pytest.mark.asyncio
async def test_proxy_registration_does_not_report_local_engine_capacity(
    monkeypatch, caplog
):
    # Keep this backend-heavy import lazy: register.py imports
    # sglang.srt.environ.envs, which is unavailable to the
    # `pytest-marker-report` collection environment.
    from dynamo.sglang import register

    server_args = SimpleNamespace(
        context_length=4096,
        disaggregation_mode=None,
        max_prefill_tokens=None,
        page_size=16,
        speculative_algorithm="NONE",
    )
    dynamo_args = register.DynamoConfig()
    dynamo_args.enable_local_indexer = False
    resolver = AsyncMock(return_value=99)
    bootstrap_resolver = Mock(return_value=(None, None))
    offloading_resolver = Mock(return_value=None)
    capacity = SimpleNamespace(
        max_num_seqs=None,
        max_num_batched_tokens=None,
        total_kv_blocks=None,
    )

    monkeypatch.setattr(register, "resolve_engine_max_num_seqs", resolver)
    monkeypatch.setattr(register, "model_card_dp_rank_bounds", lambda _: (0, 1))
    monkeypatch.setattr(register, "get_sglang_worker_group_id", lambda _: None)
    monkeypatch.setattr(register, "_get_bootstrap_info_for_config", bootstrap_resolver)
    monkeypatch.setattr(register, "_get_mooncake_runtime_data", lambda _: None)
    monkeypatch.setattr(register, "runtime_capacity", lambda *_: capacity)
    monkeypatch.setattr(
        register, "get_hicache_native_offloading_capacity", offloading_resolver
    )

    runtime_config = await register._get_runtime_config(None, server_args, dynamo_args)

    resolver.assert_not_awaited()
    bootstrap_resolver.assert_not_called()
    offloading_resolver.assert_not_called()
    assert runtime_config.engine_max_num_seqs is None
    assert runtime_config.context_length == 4096
    assert "Failed to get runtime config" not in caplog.text
    assert "Failed to compute bootstrap address" not in caplog.text


def test_spec_decode_runtime_data_uses_speculative_num_steps():
    server_args = SimpleNamespace(
        speculative_num_steps="5",
        speculative_algorithm="EAGLE",
    )

    assert get_spec_decode_runtime_data(server_args) == {
        "nextn": 5,
        "method": "EAGLE",
        "source": "backend_config",
    }


@pytest.mark.parametrize(
    "speculative_num_steps",
    [None, 0, "bad"],
)
def test_spec_decode_runtime_data_ignores_invalid_nextn(speculative_num_steps):
    server_args = SimpleNamespace(
        speculative_num_steps=speculative_num_steps,
        speculative_algorithm="EAGLE",
    )

    assert get_spec_decode_runtime_data(server_args) is None


@pytest.mark.parametrize(
    "speculative_algorithm, expected",
    [
        ("EAGLE", True),
        ("EAGLE3", True),
        ("FROZEN_KV_MTP", True),
        ("DFLASH", False),
        ("NGRAM", False),
        ("STANDALONE", False),
        ("NONE", False),
        (None, False),
        (
            "some_unregistered_algo",
            False,
        ),  # from_string raises -> guarded to False, no crash
    ],
)
def test_eagle_enabled_for_speculative_algorithm(speculative_algorithm, expected):
    # enable_eagle must equal sglang's SpeculativeAlgorithm.is_eagle() -- the SAME predicate the
    # radix cache uses to bigram-key its KV events -- so the KV-router frontend's block-hash window
    # matches the worker's events. EAGLE3 + FROZEN_KV_MTP were previously omitted -> cache-blind.
    # (NEXTN/EAGLE are normalized to EAGLE/FROZEN_KV_MTP in ServerArgs before register sees them.)
    # NOTE: import lazily. register.py does `from sglang.srt.environ import envs`, which is absent in
    # the lint/collection env of the `pytest-marker-report` pre-commit hook (unlike the sglang-free
    # `capacity` module imported at top), so a module-level import breaks that hook's collection.
    from dynamo.sglang.register import _eagle_enabled_for

    assert _eagle_enabled_for(speculative_algorithm) is expected


def test_hicache_publishes_native_offloading_capacity():
    server_args = SimpleNamespace(hicache_write_policy="write_back")
    assert get_hicache_native_offloading_capacity(
        server_args,
        {"max_total_num_tokens": 100, "hicache_host_total_tokens": 300},
    ) == {"total_tokens": 300}


@pytest.mark.parametrize(
    "value", [None, False, 0, 0.5, -1, "300", float("inf"), float("nan")]
)
def test_hicache_native_offloading_capacity_ignores_invalid_values(value):
    server_args = SimpleNamespace(hicache_write_policy="write_back")
    assert (
        get_hicache_native_offloading_capacity(
            server_args,
            {"max_total_num_tokens": 100, "hicache_host_total_tokens": value},
        )
        is None
    )


def test_hicache_requires_reported_host_capacity():
    assert (
        get_hicache_native_offloading_capacity(
            SimpleNamespace(hicache_write_policy="write_back"),
            {"max_total_num_tokens": 100},
        )
        is None
    )


@pytest.mark.parametrize(
    "policy, expected",
    [
        ("write_back", 300),
        ("write_through", 200),
        ("write_through_selective", None),
    ],
)
def test_hicache_capacity_accounts_for_write_policy(policy, expected):
    result = get_hicache_native_offloading_capacity(
        SimpleNamespace(hicache_write_policy=policy),
        {"max_total_num_tokens": 100, "hicache_host_total_tokens": 300},
    )

    assert (result or {}).get("total_tokens") == expected


def test_hicache_write_through_ignores_fully_overlapped_host_pool():
    assert (
        get_hicache_native_offloading_capacity(
            SimpleNamespace(hicache_write_policy="write_through"),
            {"max_total_num_tokens": 300, "hicache_host_total_tokens": 100},
        )
        is None
    )


@pytest.mark.asyncio
async def test_hicache_publish_failure_preserves_core_capacity(monkeypatch, caplog):
    from dynamo.sglang import register

    server_args = SimpleNamespace(
        context_length=4096,
        disaggregation_mode=None,
        hicache_write_policy="write_back",
        max_prefill_tokens=None,
        page_size=16,
        speculative_algorithm="NONE",
        speculative_num_steps=None,
    )
    dynamo_args = register.DynamoConfig()
    dynamo_args.enable_local_indexer = False
    scheduler_info = {
        "hicache_host_total_tokens": 300,
        "max_total_num_tokens": 1024,
    }
    engine = SimpleNamespace(
        _scheduler_init_result=SimpleNamespace(scheduler_infos=[scheduler_info])
    )
    capacity = SimpleNamespace(
        max_num_seqs=None,
        max_num_batched_tokens=1024,
        total_kv_blocks=64,
    )

    monkeypatch.setattr(register, "model_card_dp_rank_bounds", lambda _: (0, 1))
    monkeypatch.setattr(register, "get_sglang_worker_group_id", lambda _: None)
    monkeypatch.setattr(
        register, "_get_bootstrap_info_for_config", lambda _: (None, None)
    )
    monkeypatch.setattr(register, "_get_mooncake_runtime_data", lambda _: None)
    monkeypatch.setattr(register, "runtime_capacity", lambda *_: capacity)

    original_set = register.ModelRuntimeConfig.set_engine_specific

    def fail_hicache_publish(self, key, value):
        if key == register.NATIVE_OFFLOADING_CAPACITY_RUNTIME_KEY:
            raise RuntimeError("publish failed")
        return original_set(self, key, value)

    monkeypatch.setattr(
        register.ModelRuntimeConfig, "set_engine_specific", fail_hicache_publish
    )

    runtime_config = await register._get_runtime_config(
        engine, server_args, dynamo_args
    )

    assert runtime_config.total_kv_blocks == 64
    assert runtime_config.max_num_batched_tokens == 1024
    assert (
        "Failed to attach native offloading capacity from SGLang HiCache" in caplog.text
    )
