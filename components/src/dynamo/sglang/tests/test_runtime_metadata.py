# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace

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
    "value, expected",
    [
        (None, "mooncake"),
        ("", "mooncake"),
        ("  ", "  "),
        (" tenant-a ", " tenant-a "),
    ],
)
def test_mooncake_cluster_id_for_runtime_metadata(monkeypatch, value, expected):
    from dynamo.sglang.register import _get_mooncake_cluster_id

    if value is None:
        monkeypatch.delenv("MC_STORE_CLUSTER_ID", raising=False)
    else:
        monkeypatch.setenv("MC_STORE_CLUSTER_ID", value)

    assert _get_mooncake_cluster_id() == expected


@pytest.mark.parametrize(
    "master_server_address, value, expected",
    [
        ("etcd://etcd:2379", "not-an-int", None),
        ("redis://redis:6379", None, 0),
        ("redis://redis:6379", "7", 7),
        ("redis://redis:6379", "  +7suffix", 7),
        ("redis://redis:6379", "255", 255),
    ],
)
def test_mooncake_redis_db_index_for_runtime_metadata(
    monkeypatch, master_server_address, value, expected
):
    from dynamo.sglang.register import _get_mooncake_redis_db_index

    if value is None:
        monkeypatch.delenv("MC_REDIS_DB_INDEX", raising=False)
    else:
        monkeypatch.setenv("MC_REDIS_DB_INDEX", value)

    assert _get_mooncake_redis_db_index(master_server_address) == expected


@pytest.mark.parametrize("value", ["bad", "  ", "-1", "256"])
def test_mooncake_redis_db_index_rejects_invalid_values(monkeypatch, value):
    from dynamo.sglang.register import _get_mooncake_redis_db_index

    monkeypatch.setenv("MC_REDIS_DB_INDEX", value)

    with pytest.raises(ValueError, match="MC_REDIS_DB_INDEX"):
        _get_mooncake_redis_db_index("redis://redis:6379")


def _mooncake_server_args(extra_config=None):
    return SimpleNamespace(
        hicache_storage_backend="mooncake",
        hicache_storage_backend_extra_config=extra_config,
        hicache_mem_layout="page_first",
        page_size=256,
        tp_size=2,
        pp_size=1,
        speculative_algorithm=None,
        use_mla_backend=lambda: False,
    )


def test_mooncake_runtime_metadata_uses_env_config(monkeypatch):
    from dynamo.sglang.register import _get_mooncake_runtime_data

    monkeypatch.delenv("SGLANG_HICACHE_MOONCAKE_CONFIG_PATH", raising=False)
    monkeypatch.delenv("MOONCAKE_CLIENT", raising=False)
    monkeypatch.setenv("MOONCAKE_MASTER", "etcd://etcd-0:2379;etcd-1:2379")
    monkeypatch.setenv("MOONCAKE_PROTOCOL", "rdma")
    monkeypatch.setenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", "8gb")
    monkeypatch.setenv("MC_STORE_CLUSTER_ID", "prod-cluster")
    monkeypatch.setenv("DYN_MOONCAKE_KV_EVENTS_ENDPOINT", "tcp://mooncake-master:5557")

    runtime_data = _get_mooncake_runtime_data(_mooncake_server_args())

    assert runtime_data is not None
    assert runtime_data["master_server_address"] == ("etcd://etcd-0:2379;etcd-1:2379")
    assert runtime_data["cluster_id"] == "prod-cluster"
    assert runtime_data["kv_events_endpoint"] == "tcp://mooncake-master:5557"


def test_mooncake_runtime_metadata_publishes_redis_db_index(monkeypatch):
    from dynamo.sglang.register import _get_mooncake_runtime_data

    monkeypatch.delenv("SGLANG_HICACHE_MOONCAKE_CONFIG_PATH", raising=False)
    monkeypatch.delenv("MOONCAKE_CLIENT", raising=False)
    monkeypatch.setenv("MOONCAKE_MASTER", "redis://redis:6380")
    monkeypatch.setenv("MC_REDIS_DB_INDEX", "11")

    runtime_data = _get_mooncake_runtime_data(_mooncake_server_args())

    assert runtime_data is not None
    assert runtime_data["master_server_address"] == "redis://redis:6380"
    assert runtime_data["redis_db_index"] == 11


def test_mooncake_runtime_metadata_prefers_config_file_over_env(monkeypatch, tmp_path):
    from dynamo.sglang.register import _get_mooncake_runtime_data

    config_path = tmp_path / "mooncake.json"
    config_path.write_text(
        json.dumps({"master_server_address": "k8s://dynamo/mooncake-master"})
    )
    monkeypatch.setenv("SGLANG_HICACHE_MOONCAKE_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("MOONCAKE_MASTER", "etcd://ignored:2379")

    runtime_data = _get_mooncake_runtime_data(_mooncake_server_args())

    assert runtime_data is not None
    assert runtime_data["master_server_address"] == ("k8s://dynamo/mooncake-master")


def test_mooncake_runtime_metadata_prefers_extra_config_over_file_and_env(
    monkeypatch, tmp_path
):
    from dynamo.sglang.register import _get_mooncake_runtime_data

    config_path = tmp_path / "mooncake.json"
    config_path.write_text(
        json.dumps({"master_server_address": "k8s://ignored/file-master"})
    )
    monkeypatch.setenv("SGLANG_HICACHE_MOONCAKE_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("MOONCAKE_MASTER", "etcd://ignored:2379")
    extra_config = json.dumps(
        {
            "master_server_address": "etcd://extra-config:2379",
            "tp_lcm_size": 4,
        }
    )

    runtime_data = _get_mooncake_runtime_data(
        _mooncake_server_args(extra_config=extra_config)
    )

    assert runtime_data is not None
    assert runtime_data["master_server_address"] == "etcd://extra-config:2379"
    assert runtime_data["tp_lcm_size"] == 4


def test_mooncake_runtime_metadata_keeps_layout_when_config_resolution_fails(
    monkeypatch,
):
    from dynamo.sglang.register import _get_mooncake_runtime_data

    monkeypatch.delenv("SGLANG_HICACHE_MOONCAKE_CONFIG_PATH", raising=False)
    monkeypatch.delenv("MOONCAKE_CLIENT", raising=False)
    monkeypatch.delenv("MOONCAKE_MASTER", raising=False)
    monkeypatch.setenv("DYN_MOONCAKE_KV_EVENTS_ENDPOINT", "tcp://fallback:5557")

    runtime_data = _get_mooncake_runtime_data(_mooncake_server_args())

    assert runtime_data is not None
    assert runtime_data["page_size"] == 256
    assert runtime_data["kv_events_endpoint"] == "tcp://fallback:5557"
    assert runtime_data["master_server_address"] is None
    assert runtime_data["cluster_id"] is None
    assert runtime_data["redis_db_index"] is None


def test_mooncake_runtime_metadata_keeps_layout_when_redis_config_is_invalid(
    monkeypatch,
):
    from dynamo.sglang.register import _get_mooncake_runtime_data

    monkeypatch.delenv("SGLANG_HICACHE_MOONCAKE_CONFIG_PATH", raising=False)
    monkeypatch.delenv("MOONCAKE_CLIENT", raising=False)
    monkeypatch.setenv("MOONCAKE_MASTER", "redis://redis:6379")
    monkeypatch.setenv("MC_REDIS_DB_INDEX", "invalid")

    runtime_data = _get_mooncake_runtime_data(_mooncake_server_args())

    assert runtime_data is not None
    assert runtime_data["page_size"] == 256
    assert runtime_data["master_server_address"] is None
    assert runtime_data["cluster_id"] is None
    assert runtime_data["redis_db_index"] is None


def test_mooncake_runtime_metadata_keeps_layout_without_config_import(monkeypatch):
    from dynamo.sglang import register

    monkeypatch.setattr(register, "MooncakeStoreConfig", None)
    monkeypatch.setattr(
        register, "_MOONCAKE_CONFIG_IMPORT_ERROR", "unavailable in test"
    )

    runtime_data = register._get_mooncake_runtime_data(_mooncake_server_args())

    assert runtime_data is not None
    assert runtime_data["page_size"] == 256
    assert runtime_data["master_server_address"] is None
    assert runtime_data["cluster_id"] is None
    assert runtime_data["redis_db_index"] is None


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


def test_runtime_config_without_engine_omits_token_budget(monkeypatch, caplog):
    from dynamo.sglang import register

    server_args = SimpleNamespace(
        allow_auto_truncate=False,
        context_length=4096,
        disaggregation_mode=None,
        max_prefill_tokens=None,
        page_size=16,
        speculative_algorithm="NONE",
        speculative_num_steps=None,
    )
    dynamo_args = register.DynamoConfig()
    dynamo_args.enable_local_indexer = False
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

    runtime_config = asyncio.run(
        register.get_runtime_config(None, server_args, dynamo_args)
    )

    assert TOKEN_BUDGET_RUNTIME_KEY not in runtime_config.runtime_data
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
