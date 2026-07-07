# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from dynamo.sglang.capacity import get_spec_decode_runtime_data

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


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


def test_hicache_capacity_runtime_data_uses_scheduler_capacity():
    from dynamo.sglang.register import _get_hicache_capacity_runtime_data

    assert _get_hicache_capacity_runtime_data({"hicache_host_total_tokens": 300}) == {
        "host_total_tokens": 300
    }


@pytest.mark.parametrize("value", [None, 0, -1, "300"])
def test_hicache_capacity_runtime_data_ignores_invalid_values(value):
    from dynamo.sglang.register import _get_hicache_capacity_runtime_data

    assert (
        _get_hicache_capacity_runtime_data({"hicache_host_total_tokens": value}) is None
    )


@pytest.mark.asyncio
async def test_hicache_publish_failure_preserves_core_capacity(monkeypatch, caplog):
    from dynamo.sglang import register

    server_args = SimpleNamespace(
        context_length=4096,
        disaggregation_mode=None,
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
        if key == register.SGLANG_HICACHE_CAPACITY_RUNTIME_KEY:
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
    assert "Failed to attach SGLang HiCache capacity runtime metadata" in caplog.text
