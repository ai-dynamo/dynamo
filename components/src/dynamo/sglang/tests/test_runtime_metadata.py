# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

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


@pytest.mark.asyncio
async def test_engine_sequence_capacity_uses_smallest_effective_local_rank():
    engine = SimpleNamespace(
        tokenizer_manager=SimpleNamespace(
            get_internal_state=AsyncMock(
                return_value=[
                    {"effective_max_running_requests_per_dp": 7},
                    {"effective_max_running_requests_per_dp": True},
                    {"effective_max_running_requests_per_dp": 5},
                ]
            )
        )
    )

    capacity = await resolve_engine_max_num_seqs(
        engine,
        SimpleNamespace(max_running_requests=999, dp_size=1),
        local_dp_size=3,
    )

    assert capacity == 15


@pytest.mark.asyncio
async def test_engine_sequence_capacity_falls_back_to_configured_per_rank(caplog):
    engine = SimpleNamespace(
        tokenizer_manager=SimpleNamespace(
            get_internal_state=AsyncMock(side_effect=RuntimeError("not supported"))
        )
    )

    capacity = await resolve_engine_max_num_seqs(
        engine,
        SimpleNamespace(max_running_requests=24, dp_size=4),
        local_dp_size=2,
    )

    assert capacity == 12
    assert "falling back to configured max_running_requests" in caplog.text


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
async def test_proxy_registration_does_not_report_local_engine_capacity(monkeypatch):
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
    capacity = SimpleNamespace(
        max_num_seqs=None,
        max_num_batched_tokens=None,
        total_kv_blocks=None,
    )

    monkeypatch.setattr(register, "resolve_engine_max_num_seqs", resolver)
    monkeypatch.setattr(register, "model_card_dp_rank_bounds", lambda _: (0, 1))
    monkeypatch.setattr(register, "get_sglang_worker_group_id", lambda _: None)
    monkeypatch.setattr(
        register, "_get_bootstrap_info_for_config", lambda _: (None, None)
    )
    monkeypatch.setattr(register, "_get_mooncake_runtime_data", lambda _: None)
    monkeypatch.setattr(register, "runtime_capacity", lambda *_: capacity)

    runtime_config = await register._get_runtime_config(None, server_args, dynamo_args)

    resolver.assert_not_awaited()
    assert runtime_config.engine_max_num_seqs is None


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
