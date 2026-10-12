# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dynamo.common.token_budget import TOKEN_BUDGET_RUNTIME_KEY, TokenBudget
from dynamo.sglang.capacity import (
    get_hicache_native_offloading_capacity,
    get_spec_decode_runtime_data,
    kv_event_block_size,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.parametrize(
    "parser, skip_tokenizer_init, has_engine, has_tokenizer, supports_gate, think_end_ids, expected",
    [
        ("kimi_k3", False, True, True, True, [163588, 39964, 163589], True),
        (None, False, True, True, True, [163588], False),
        ("kimi_k3", True, True, True, True, [163588], False),
        ("kimi_k3", False, False, True, True, [163588], False),
        ("kimi_k3", False, True, False, True, [163588], False),
        ("kimi_k3", False, True, True, False, [163588], False),
        ("kimi_k3", False, True, True, True, [], False),
    ],
)
def test_structural_tag_reasoning_policy_advertises_only_an_available_gate(
    parser,
    skip_tokenizer_init,
    has_engine,
    has_tokenizer,
    supports_gate,
    think_end_ids,
    expected,
):
    from dynamo.sglang import register

    async def async_generate(*, require_reasoning=False):
        pass

    async def legacy_async_generate(*, sampling_params=None):
        pass

    tokenizer = Mock()
    tokenizer.encode.return_value = think_end_ids
    engine = (
        SimpleNamespace(
            tokenizer_manager=SimpleNamespace(
                tokenizer=tokenizer if has_tokenizer else None
            ),
            async_generate=async_generate if supports_gate else legacy_async_generate,
        )
        if has_engine
        else None
    )
    server_args = SimpleNamespace(
        reasoning_parser=parser, skip_tokenizer_init=skip_tokenizer_init
    )
    runtime_config = register.ModelRuntimeConfig()

    register.publish_sglang_structural_tag_reasoning_policy(
        runtime_config, engine, server_args
    )

    key = register.TOOL_CALL_STRUCTURAL_TAG_REASONING_GATE_RUNTIME_KEY
    assert (key in runtime_config.runtime_data) is expected
    if expected:
        assert json.loads(runtime_config.runtime_data[key]) is True


@pytest.mark.parametrize(
    "server_args, expected",
    [
        (SimpleNamespace(page_size=64), 64),
        (SimpleNamespace(page_size=64, dcp_size=1), 64),
        (SimpleNamespace(page_size=64, dcp_size=None), 64),
        (SimpleNamespace(page_size=64, dcp_size=8), 512),
    ],
)
def test_kv_event_block_size_accounts_for_dcp(server_args, expected):
    assert kv_event_block_size(server_args) == expected


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


@pytest.mark.parametrize(
    "allow_auto_truncate, validate_total_tokens, reserved_tokens, expected",
    [
        (
            False,
            True,
            4,
            TokenBudget(252, True, True),
        ),
        (
            True,
            True,
            0,
            TokenBudget(256, False, False),
        ),
        (
            False,
            False,
            0,
            TokenBudget(256, True, False),
        ),
    ],
)
def test_token_budget_matches_sglang_policy(
    allow_auto_truncate, validate_total_tokens, reserved_tokens, expected
):
    from dynamo.sglang.register import _get_token_budget

    engine = SimpleNamespace(
        tokenizer_manager=SimpleNamespace(
            context_len=256,
            validate_total_tokens=validate_total_tokens,
            num_reserved_tokens=reserved_tokens,
        )
    )
    server_args = SimpleNamespace(
        context_length=None,
        allow_auto_truncate=allow_auto_truncate,
    )

    assert _get_token_budget(engine, server_args) == expected


@pytest.mark.parametrize(
    "disaggregation_mode, runtime_supported, enable_multimodal, capability_expected",
    [
        (None, True, False, False),
        ("null", True, False, False),
        ("prefill", False, False, False),
        ("prefill", True, True, False),
        ("prefill", True, False, True),
        ("decode", True, False, True),
    ],
)
def test_runtime_config_publishes_supported_disagg_capabilities(
    monkeypatch,
    caplog,
    disaggregation_mode,
    runtime_supported,
    enable_multimodal,
    capability_expected,
):
    from dynamo.sglang import register

    server_args = SimpleNamespace(
        allow_auto_truncate=False,
        context_length=4096,
        disaggregation_mode=disaggregation_mode,
        max_prefill_tokens=None,
        page_size=16,
        reasoning_parser="kimi_k3",
        skip_tokenizer_init=False,
        speculative_algorithm="NONE",
        speculative_num_steps=None,
    )
    dynamo_args = register.DynamoConfig()
    dynamo_args.enable_local_indexer = False
    dynamo_args.enable_multimodal = enable_multimodal
    capacity = SimpleNamespace(
        max_num_seqs=None,
        max_num_batched_tokens=None,
        total_kv_blocks=None,
    )

    monkeypatch.setattr(register, "model_card_dp_rank_bounds", lambda _: (0, 1))
    monkeypatch.setattr(register, "get_sglang_worker_group_id", lambda _: None)
    monkeypatch.setattr(register, "apply_topology_config", lambda _: None)
    monkeypatch.setattr(
        register, "_get_bootstrap_info_for_config", lambda _: (None, None)
    )
    monkeypatch.setattr(register, "get_spec_decode_runtime_data", lambda _: None)
    monkeypatch.setattr(register, "_get_mooncake_runtime_data", lambda _: None)
    monkeypatch.setattr(register, "runtime_capacity", lambda *_: capacity)
    monkeypatch.setattr(
        register, "supports_disagg_prefill_cancel_anytime", lambda _: runtime_supported
    )
    engine = None
    if runtime_supported:

        async def async_generate(*, require_reasoning=False):
            pass

        tokenizer = Mock()
        tokenizer.encode.return_value = [163588, 39964, 163589]
        engine = SimpleNamespace(
            async_generate=async_generate,
            tokenizer_manager=SimpleNamespace(
                tokenizer=tokenizer,
                context_len=4096,
                validate_total_tokens=True,
                num_reserved_tokens=0,
                rid_to_state={},
            ),
            _scheduler_init_result=SimpleNamespace(scheduler_infos=[{}]),
        )

    runtime_config = asyncio.run(
        register.get_runtime_config(engine, server_args, dynamo_args)
    )

    assert (
        TOKEN_BUDGET_RUNTIME_KEY in runtime_config.runtime_data
    ) is runtime_supported
    capability = register.DISAGG_PREFILL_CANCEL_ANYTIME_V1
    assert (capability in runtime_config.runtime_data) is capability_expected
    if capability_expected:
        assert json.loads(runtime_config.runtime_data[capability]) is True
    reasoning_gate = register.TOOL_CALL_STRUCTURAL_TAG_REASONING_GATE_RUNTIME_KEY
    assert (reasoning_gate in runtime_config.runtime_data) is (
        runtime_supported and not enable_multimodal
    )
    assert "Failed to get runtime config" not in caplog.text


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


def test_hicache_derives_ratio_based_capacity():
    assert get_hicache_native_offloading_capacity(
        SimpleNamespace(
            enable_hierarchical_cache=True,
            hicache_size=0,
            hicache_write_policy="write_back",
            hicache_ratio=3.0,
            page_size=16,
        ),
        {"max_total_num_tokens": 100},
    ) == {"total_tokens": 304}


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
        allow_auto_truncate=False,
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
        _scheduler_init_result=SimpleNamespace(scheduler_infos=[scheduler_info]),
        tokenizer_manager=SimpleNamespace(
            context_len=4096,
            validate_total_tokens=True,
            num_reserved_tokens=4,
        ),
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

    runtime_config = await register.get_runtime_config(engine, server_args, dynamo_args)

    assert runtime_config.total_kv_blocks == 64
    assert runtime_config.max_num_batched_tokens == 1024
    assert json.loads(runtime_config.runtime_data[TOKEN_BUDGET_RUNTIME_KEY]) == {
        "combined_limit": 4092,
        "reject_prompt_overflow": True,
        "reject_total_overflow": True,
    }
    assert (
        "Failed to attach native offloading capacity from SGLang HiCache" in caplog.text
    )
