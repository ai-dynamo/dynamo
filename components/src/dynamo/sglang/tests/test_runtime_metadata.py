# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.common.token_budget import TOKEN_BUDGET_RUNTIME_KEY, TokenBudget
from dynamo.sglang.capacity import (
    get_hicache_native_offloading_capacity,
    get_spec_decode_runtime_data,
    kv_event_block_size,
)

# SGLang may be absent or partially stubbed during marker-only collection.
# Keep import errors so the fixture still fails if a registration test runs.
_register_import_error = None
try:
    from dynamo.sglang import register as _register
except ImportError as exc:
    _register = None
    _register_import_error = exc

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.fixture
def register():
    """Provide registration code only to tests that require SGLang."""
    if _register_import_error is not None:
        raise _register_import_error
    return _register


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
    register,
    parser,
    skip_tokenizer_init,
    has_engine,
    has_tokenizer,
    supports_gate,
    think_end_ids,
    expected,
):
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
    register,
    monkeypatch,
    caplog,
    disaggregation_mode,
    runtime_supported,
    enable_multimodal,
    capability_expected,
):
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
async def test_hicache_publish_failure_preserves_core_capacity(
    register, monkeypatch, caplog
):
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


# Expected values follow SGLang MooncakeStore's `config_prefix`: backend tag (when
# not None) then served model name with "/" -> "-", joined with "_".
@pytest.mark.parametrize(
    "extra_config, served_model_name, expected",
    [
        ({}, "Qwen/Qwen3-0.6B", "Qwen-Qwen3-0.6B"),
        ({"extra_backend_tag": "t"}, "Qwen/Qwen3-0.6B", "t_Qwen-Qwen3-0.6B"),
        ({"extra_backend_tag": ""}, "Qwen/Qwen3-0.6B", "_Qwen-Qwen3-0.6B"),
        ({"extra_backend_tag": 7}, "Qwen/Qwen3-0.6B", "7_Qwen-Qwen3-0.6B"),
        ({}, "my-model", "my-model"),
        ({"extra_backend_tag": "t"}, None, "t"),
        ({}, None, None),
    ],
)
def test_mooncake_runtime_data_publishes_sglang_key_prefix(
    register, monkeypatch, extra_config, served_model_name, expected
):
    monkeypatch.setattr(register, "sglang_uses_mla_backend", lambda _: False)
    server_args = SimpleNamespace(
        hicache_storage_backend="mooncake",
        hicache_storage_backend_extra_config=json.dumps(extra_config),
        served_model_name=served_model_name,
        page_size=64,
        tp_size=1,
        pp_size=1,
        speculative_algorithm=None,
    )

    runtime_data = register._get_mooncake_runtime_data(server_args)

    assert runtime_data["key_prefix"] == expected


@pytest.mark.parametrize(
    "filename, content",
    [
        ("extra.json", '{"extra_backend_tag": "équipe"}'),
        ("extra.toml", 'extra_backend_tag = "équipe"\n'),
        ("extra.yaml", "extra_backend_tag: équipe\n"),
        ("extra.yml", "extra_backend_tag: équipe\n"),
    ],
)
def test_mooncake_key_prefix_reads_at_file_extra_config(
    register, monkeypatch, tmp_path, filename, content
):
    """Preserve UTF-8 cache tags even when text reads default to ASCII."""

    def open_with_ascii_default(path, mode="r", **kwargs):
        if "b" not in mode:
            kwargs.setdefault("encoding", "ascii")
        return open(path, mode, **kwargs)

    config_path = tmp_path / filename
    config_path.write_bytes(content.encode("utf-8"))
    monkeypatch.setattr(register, "open", open_with_ascii_default, raising=False)
    monkeypatch.setattr(register, "sglang_uses_mla_backend", lambda _: False)
    server_args = SimpleNamespace(
        hicache_storage_backend="mooncake",
        hicache_storage_backend_extra_config=f"@{config_path}",
        served_model_name="Qwen/Qwen3-0.6B",
        page_size=64,
        tp_size=1,
        pp_size=1,
        speculative_algorithm=None,
    )

    runtime_data = register._get_mooncake_runtime_data(server_args)

    assert runtime_data["extra_backend_tag"] == "équipe"
    assert runtime_data["key_prefix"] == "équipe_Qwen-Qwen3-0.6B"


@pytest.mark.parametrize("filename", ["missing.json", "extra.ini"])
def test_unreadable_at_file_extra_config_is_ignored(register, tmp_path, filename):
    path = tmp_path / filename
    if filename.endswith(".ini"):
        path.write_text("extra_backend_tag = t\n")

    assert register._parse_hicache_storage_extra_config(f"@{path}") == {}


@pytest.fixture
def hicache_layout_envelope():
    def pool(name, policy="all_pages", **kwargs):
        return {
            "pool": name,
            "suffixes": [f"__{name}"],
            "hit_policy": policy,
            **kwargs,
        }

    return {
        "enabled": True,
        "backend": "mooncake",
        "page_size": 16,
        "hicache_object_layout": {
            "key_prefix": "private-prefix",
            "is_mla_backend": True,
            "pp_rank": 0,
            "pp_size": 1,
            "pools": [
                pool("swa", "trailing_pages", trailing_pages=4),
                pool("compress8"),
                pool("indexer"),
                pool("draft_swa", "trailing_pages", trailing_pages=4),
            ],
        },
    }


@pytest.fixture
def mooncake_registration(monkeypatch, hicache_layout_envelope):
    # As above, defer backend imports for marker-only collection.
    from dynamo.sglang import register

    monkeypatch.setenv("DYN_TOPOLOGY_ENABLED", "false")
    monkeypatch.delenv("DYN_MOONCAKE_KV_EVENTS_ENDPOINT", raising=False)
    monkeypatch.setattr(
        register, "sglang_uses_mla_backend", lambda args: args.is_mla_model
    )
    monkeypatch.setattr(
        register, "_get_bootstrap_info_for_config", lambda _: (None, None)
    )
    submit = AsyncMock()
    monkeypatch.setattr(register, "register_model", submit)
    server_args = SimpleNamespace(
        allow_auto_truncate=False,
        context_length=4096,
        disaggregation_mode=None,
        page_size=16,
        tp_size=8,
        pp_size=1,
        dp_size=2,
        speculative_algorithm="EAGLE3",
        speculative_num_steps=3,
        hicache_storage_backend="mooncake",
        hicache_mem_layout="page_head",
        hicache_storage_backend_extra_config={
            "tp_lcm_size": 16,
            "extra_backend_tag": "tag",
            "kv_events_endpoint": "ipc:///unused-mooncake-events",
        },
        is_mla_model=True,
        model_path="unused-model",
        served_model_name="unused-model",
    )
    info = {
        "max_total_num_tokens": 1024,
        "hicache_object_layout": hicache_layout_envelope,
    }
    engine = SimpleNamespace(
        _scheduler_init_result=SimpleNamespace(scheduler_infos=[info, dict(info)]),
        tokenizer_manager=SimpleNamespace(
            context_len=4096,
            validate_total_tokens=True,
            num_reserved_tokens=4,
        ),
        shutdown=Mock(),
    )
    dynamo_args = register.DynamoConfig()
    dynamo_args.enable_local_indexer = False
    dynamo_args.use_sglang_tokenizer = False

    def run(with_engine=True):
        ready = asyncio.Event()
        asyncio.run(
            register.register_model_with_readiness_gate(
                engine if with_engine else None,
                Mock(),
                server_args,
                dynamo_args,
                readiness_gate=ready,
                worker_type=register.WorkerType.Aggregated,
            )
        )
        assert ready.is_set()
        engine.shutdown.assert_not_called()
        config = submit.await_args.kwargs["runtime_config"]
        assert isinstance(config, register.ModelRuntimeConfig)
        assert config.data_parallel_size == 2
        if with_engine:
            assert config.total_kv_blocks == 64
        return config

    return SimpleNamespace(run=run, engine=engine, server_args=server_args)


@pytest.mark.parametrize("is_mla_model", [True, False])
@pytest.mark.parametrize("key_prefix", [None, "private-prefix"])
@pytest.mark.parametrize("string_config", [False, True])
def test_registration_publishes_physical_mooncake_layout(
    mooncake_registration,
    hicache_layout_envelope,
    is_mla_model,
    key_prefix,
    string_config,
):
    # DSV4 is_mla_model=False still has rank-replicated physical pools.
    args = mooncake_registration.server_args
    args.is_mla_model = is_mla_model
    hicache_layout_envelope["hicache_object_layout"]["key_prefix"] = key_prefix
    if string_config:
        args.hicache_storage_backend_extra_config = json.dumps(
            args.hicache_storage_backend_extra_config
        )
    config = mooncake_registration.run()
    assert json.loads(config.runtime_data["sglang_hicache_mooncake"]) == {
        "backend": "mooncake",
        "tenant_id": "default",
        "page_size": 16,
        "tp_size": 8,
        "pp_size": 1,
        "is_mla_model": is_mla_model,
        "is_eagle": True,
        "tp_lcm_size": 16,
        "should_split_heads": False,
        "extra_backend_tag": "tag",
        "key_prefix": "tag_unused-model",
        "kv_events_endpoint": "ipc:///unused-mooncake-events",
        "hicache_object_layout": hicache_layout_envelope["hicache_object_layout"],
    }


@pytest.mark.parametrize("with_engine", [True, False])
def test_registration_retains_legacy_mooncake_metadata(
    mooncake_registration, with_engine
):
    for info in mooncake_registration.engine._scheduler_init_result.scheduler_infos:
        info.pop("hicache_object_layout")
    config = mooncake_registration.run(with_engine=with_engine)
    metadata = json.loads(config.runtime_data["sglang_hicache_mooncake"])
    assert "hicache_object_layout" not in metadata
    assert metadata["backend"] == "mooncake"
    assert metadata["tp_size"] == 8
    assert metadata["is_mla_model"] is True


def test_registration_ignores_non_finite_tp_lcm_size(mooncake_registration):
    for info in mooncake_registration.engine._scheduler_init_result.scheduler_infos:
        info.pop("hicache_object_layout")
    extra = mooncake_registration.server_args.hicache_storage_backend_extra_config
    extra["tp_lcm_size"] = float("inf")
    config = mooncake_registration.run()
    metadata = json.loads(config.runtime_data["sglang_hicache_mooncake"])
    assert metadata["tp_lcm_size"] is None


@pytest.mark.parametrize("pp_rank, published", [(None, True), (0, False), (1, False)])
def test_registration_publishes_layout_covering_all_pp_stages(
    mooncake_registration, hicache_layout_envelope, pp_rank, published
):
    # Only a layout merged across stages (pp_rank null) can cover them all.
    mooncake_registration.server_args.pp_size = 2
    layout = hicache_layout_envelope["hicache_object_layout"]
    layout["pp_size"], layout["pp_rank"] = 2, pp_rank
    config = mooncake_registration.run()
    if not published:
        assert "sglang_hicache_mooncake" not in config.runtime_data
        return
    metadata = json.loads(config.runtime_data["sglang_hicache_mooncake"])
    assert metadata["pp_size"] == 2
    assert metadata["hicache_object_layout"] == layout


def test_registration_accepts_uint32_trailing_pages_limit(
    mooncake_registration, hicache_layout_envelope
):
    layout = hicache_layout_envelope["hicache_object_layout"]
    layout["pools"][0]["trailing_pages"] = 2**32 - 1
    config = mooncake_registration.run()
    metadata = json.loads(config.runtime_data["sglang_hicache_mooncake"])
    assert metadata["hicache_object_layout"] == layout


@pytest.mark.parametrize("tp_lcm_size", [0, 2**32])
def test_registration_ignores_invalid_tp_lcm_size_with_layout(
    mooncake_registration, hicache_layout_envelope, tp_lcm_size
):
    args = mooncake_registration.server_args
    args.hicache_storage_backend_extra_config["tp_lcm_size"] = tp_lcm_size
    config = mooncake_registration.run()
    metadata = json.loads(config.runtime_data["sglang_hicache_mooncake"])
    assert metadata["tp_lcm_size"] is None
    assert (
        metadata["hicache_object_layout"]
        == hicache_layout_envelope["hicache_object_layout"]
    )


def test_registration_rejects_null_window_on_all_pages_pool(
    mooncake_registration, hicache_layout_envelope
):
    hicache_layout_envelope["hicache_object_layout"]["pools"][1][
        "trailing_pages"
    ] = None
    config = mooncake_registration.run()
    assert "sglang_hicache_mooncake" not in config.runtime_data


@pytest.mark.parametrize(
    "envelope", [None, [], {}, {"enabled": False, "reason": "private-reason"}]
)
def test_registration_disables_l3_without_failing_worker(
    mooncake_registration, envelope, caplog
):
    for info in mooncake_registration.engine._scheduler_init_result.scheduler_infos:
        info["hicache_object_layout"] = envelope
    with caplog.at_level("DEBUG"):
        config = mooncake_registration.run()
    assert "sglang_hicache_mooncake" not in config.runtime_data
    assert "disabling L3 lookup" in caplog.text
    assert "private-reason" not in caplog.text
    if isinstance(envelope, dict) and envelope.get("enabled") is False:
        assert not any(record.levelname == "WARNING" for record in caplog.records)


def _disabled_envelope(**layout):
    return {
        "enabled": False,
        "backend": "mooncake",
        "page_size": 16,
        "hicache_object_layout": {"pools": [], "pp_rank": 0, **layout},
        "reason": "private-reason",
    }


@pytest.mark.parametrize(
    "layout",
    [{"pp_size": 1, "is_mla_backend": False}, {"pp_size": 2, "is_mla_backend": True}],
)
def test_registration_keeps_legacy_metadata_for_rank_local_layouts(
    mooncake_registration, layout
):
    infos = mooncake_registration.engine._scheduler_init_result.scheduler_infos
    for info in infos:
        info["hicache_object_layout"] = _disabled_envelope(**layout)
    config = mooncake_registration.run()
    metadata = json.loads(config.runtime_data["sglang_hicache_mooncake"])
    assert metadata["backend"] == "mooncake"
    assert "hicache_object_layout" not in metadata


@pytest.mark.parametrize("mixed_with", ["enabled", "auxiliary_disabled"])
def test_registration_disables_l3_for_mixed_disabled_layouts(
    mooncake_registration, hicache_layout_envelope, mixed_with
):
    infos = mooncake_registration.engine._scheduler_init_result.scheduler_infos
    infos[0]["hicache_object_layout"] = _disabled_envelope(
        pp_size=1, is_mla_backend=False
    )
    infos[1]["hicache_object_layout"] = (
        hicache_layout_envelope
        if mixed_with == "enabled"
        else _disabled_envelope(pp_size=1, is_mla_backend=True)
    )
    config = mooncake_registration.run()
    assert "sglang_hicache_mooncake" not in config.runtime_data


@pytest.mark.parametrize(
    "field, value",
    [
        ("enabled", 1),
        ("backend", "other"),
        ("page_size", 0),
        ("page_size", True),
        ("page_size", 32),
        ("page_size", 2**32),
        ("page_size", None),
        ("page_size", "16"),
        ("key_prefix", 7),
        ("pp_rank", -1),
        ("pp_rank", 1),
        ("pp_size", 2),
        ("is_mla_backend", False),
        ("pools", []),
        ("pool", ""),
        ("suffixes", []),
        ("suffixes", [None]),
        ("hit_policy", "private-policy"),
        ("trailing_pages", 0),
        ("trailing_pages", None),
        ("trailing_pages", True),
        ("trailing_pages", 2**32),
        ("trailing_pages", 2**64 - 1),
    ],
)
def test_registration_rejects_invalid_layout_without_legacy_fallback(
    mooncake_registration, hicache_layout_envelope, field, value, caplog
):
    layout = hicache_layout_envelope["hicache_object_layout"]
    if field in hicache_layout_envelope:
        parent = hicache_layout_envelope
    elif field in layout:
        parent = layout
    else:
        parent = layout["pools"][0]
    parent[field] = value
    config = mooncake_registration.run()
    assert "sglang_hicache_mooncake" not in config.runtime_data
    assert "disabling L3 lookup" in caplog.text
    assert "private-prefix" not in caplog.text
    assert "private-policy" not in caplog.text


@pytest.mark.parametrize(
    "mode", ["missing_window", "missing_rank", "inconsistent", "partly_missing"]
)
def test_registration_requires_complete_consistent_layouts(
    mooncake_registration, hicache_layout_envelope, mode, caplog
):
    infos = mooncake_registration.engine._scheduler_init_result.scheduler_infos
    layout = hicache_layout_envelope["hicache_object_layout"]
    if mode == "missing_window":
        layout["pools"][0].pop("trailing_pages")
    elif mode == "missing_rank":
        layout.pop("pp_rank")
    elif mode == "inconsistent":
        infos[1]["hicache_object_layout"] = deepcopy(hicache_layout_envelope)
        other = infos[1]["hicache_object_layout"]["hicache_object_layout"]
        other["key_prefix"] = "private-other"
    else:
        infos[1].pop("hicache_object_layout")
    config = mooncake_registration.run()
    assert "sglang_hicache_mooncake" not in config.runtime_data
    assert "disabling L3 lookup" in caplog.text
    assert "private-other" not in caplog.text


@pytest.mark.parametrize(
    "extra_config, file_tenant, env_tenant, expected",
    [
        (
            {"master_server_address": "master", "tenant_id": " tenant-a "},
            "file",
            "env",
            "tenant-a",
        ),
        ({"tenant_id": "unused"}, "file", "env", "file"),
        (
            {"master_server_address": "master", "tenant_id": None},
            "file",
            "env",
            "default",
        ),
        ({}, None, "  ", "default"),
    ],
)
def test_mooncake_runtime_data_publishes_effective_tenant(
    register, monkeypatch, tmp_path, extra_config, file_tenant, env_tenant, expected
):
    monkeypatch.setattr(register, "envs", SimpleNamespace(MOONCAKE_TENANT_ID=object()))
    monkeypatch.setattr(register, "sglang_uses_mla_backend", lambda _: False)
    monkeypatch.delenv("SGLANG_HICACHE_MOONCAKE_CONFIG_PATH", raising=False)
    monkeypatch.delenv("MOONCAKE_TENANT_ID", raising=False)
    if file_tenant is not None:
        config_path = tmp_path / "mooncake.json"
        config_path.write_text(json.dumps({"tenant_id": file_tenant}))
        monkeypatch.setenv("SGLANG_HICACHE_MOONCAKE_CONFIG_PATH", str(config_path))
    if env_tenant is not None:
        monkeypatch.setenv("MOONCAKE_TENANT_ID", env_tenant)
    args = SimpleNamespace(
        hicache_storage_backend="mooncake",
        hicache_storage_backend_extra_config=json.dumps(extra_config),
        served_model_name="model",
        page_size=4,
    )
    assert register._get_mooncake_runtime_data(args)["tenant_id"] == expected


def test_mooncake_tenant_stays_default_without_upstream_support(register, monkeypatch):
    monkeypatch.setattr(register, "envs", SimpleNamespace())
    monkeypatch.setenv("MOONCAKE_TENANT_ID", "tenant-a")
    assert (
        register._mooncake_tenant_id(
            {"master_server_address": "master", "tenant_id": "tenant-a"}
        )
        == "default"
    )
